from webtools import app
import arrow
from .models import GenderSample, GenderUserAnnotation, LearnedModel, LearningTask
from .models import AccuracyMetric, GenderSampleResult
from sqlalchemy import func, or_, and_, desc, not_
from webtools.utils import  send_slack_message
from .utils import clear_old_tasks, check_working_tasks, get_learned_models_count, get_finished_time_task
import celery
from .celery_tasks import test_on_error, test_on_success, run_test, run_train, train_on_error, train_on_success
from .celery_tasks import run_train_k_folds, train_k_folds_on_error, train_k_folds_on_success
from .celery_tasks import run_test_k_folds, test_k_folds_on_error, test_k_folds_on_success
from .celery_tasks import update_gender_cv_partition
from .celery_tasks import run_deploy, deploy_on_error, deploy_on_success

import os.path
import os
from shutil import copyfile

@app.celery.task()
def annotation_statistics():
    utc = arrow.utcnow()
    expected_utc = utc.shift(hours=-24)

    base_query = app.db.session.query(GenderSample).\
        filter(GenderSample.changed_ts>expected_utc).\
        join(GenderUserAnnotation)

    total_annotated_last = base_query.count()
    test_annotated_last = base_query.filter(GenderSample.always_test==True).count()
    train_annotated_last = base_query.filter(GenderSample.always_test==False).count()

    print('annotation statistics: {} annotated, {} for train, {} for test'.format(total_annotated_last,
                                                                                  train_annotated_last,
                                                                                  test_annotated_last))
    msg = ':memo: Daily annotation statistics\n'
    msg += '*Gender:*\n'
    if total_annotated_last == 0:
        msg += ' No annotated samples :waiting:'.format(total_annotated_last)
    else:
        msg += ' Total samples annotated: {}\n'.format(total_annotated_last)
        msg += ' For training: {}\n'.format(test_annotated_last)
        msg += ' For testing: {}'.format(train_annotated_last)

    send_slack_message(msg)

@app.celery.task()
def auto_training():
    problem_name = 'gender'
    problem_type = 'train'

    if check_working_tasks(problem_name, problem_type):
        print('auto-training: training already running, skip')
        return

    trigger_train = False
    reason = ''

    # check time
    if not trigger_train:
        finished_ts = get_finished_time_task(problem_name, problem_type)
        min_hours = app.config.get('TRIGGER_TRAIN_MIN_HOURS')
        print('min_hours: {}'.format(min_hours))
        if finished_ts is not None:
            trigger_min_ts = finished_ts.shift(hours=min_hours)
            if arrow.utcnow() < trigger_min_ts:
                print('auto-training: exit on time constraint')
                return

    if get_learned_models_count(problem_name) == 0:
        reason = 'first model'
        trigger_train = True

    # number of new samples (new annotated or added from script)
    n_new = 0
    if not trigger_train:

        n_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.is_hard == False,
                        GenderSample.is_bad == False,
                        GenderSample.always_test == False)). \
            outerjoin(GenderUserAnnotation). \
            filter(or_(and_(GenderUserAnnotation.id == None,
                            GenderSample.is_annotated_gt),
                       and_(GenderUserAnnotation.id != None,
                            GenderUserAnnotation.is_hard == False,
                            GenderUserAnnotation.is_bad == False))).count()

        model = LearnedModel.query.filter(and_(LearnedModel.problem_name == problem_name,
                                               LearnedModel.k_fold == None,
                                               LearnedModel.finished_ts != None)).first()
        if model:
            n_new = n_samples - model.num_samples
            min_samples = app.config.get('TRIGGER_TRAIN_MIN_SAMPLES')
            if n_new >= min_samples:
                trigger_train = True
                reason = '{} new samples'.format(n_new)

    # number of changed samples (annotated new or changed old samples through web-interface)
    n_changed_samples = 0
    if not trigger_train:
        n_changed_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.always_test == False,
                        GenderSample.is_checked == True,
                        GenderSample.is_changed == True)).count()

        min_samples = app.config.get('TRIGGER_TRAIN_MIN_SAMPLES')
        if n_changed_samples >= min_samples:
            trigger_train = True
            reason = '{} samples have been annotated'.format(n_changed_samples)

    # train if n_changed > 0 and time constraint is satisfied
    if not trigger_train and (n_changed_samples > 0 or n_new > 0):
        finished_ts = get_finished_time_task(problem_name, problem_type)

        if finished_ts is not None:
            last_train_ts = finished_ts
            trigger_max_hours = app.config.get('TRIGGER_TRAIN_MAX_HOURS')
            trigger_train_ts = last_train_ts.shift(hours=trigger_max_hours)

            if arrow.utcnow() >= trigger_train_ts:
                trigger_train = True
                reason = 'time threshold, changed samples: {}, new samples: {}'.format(n_changed_samples, n_new)

    if trigger_train:
        print('run auto-training: {}'.format(reason))

        clear_old_tasks(problem_name, problem_type)

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name, problem_type=problem_type,
                               task_id=task_id, started_ts=utc)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_train.apply_async((problem_name,), task_id=task_id,
                                     link_error=train_on_error.s(), link=train_on_success.s(),
                                     queue='learning')

        print('{} task successfully started'.format(task.id))

        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += '{}, run training'.format(reason)

        send_slack_message(msg)
    else:
        print('auto-training: exit')

@app.celery.task()
def auto_training_k_folds():
    print('auto-training k-folds')

    update_gender_cv_partition()

    problem_name = 'gender'
    problem_type = 'train_k_folds'

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    folds_tested = []
    for k_fold in range(k_folds):

        trigger_train = False
        reason = ''

        if check_working_tasks(problem_name, problem_type, k_fold=k_fold):
            print('k-fold {}: attempted to start training while other task not finished'.format(k_fold))
            continue

        # check time
        if not trigger_train:
            finished_ts = get_finished_time_task(problem_name, problem_type, k_fold=k_fold)
            min_hours = app.config.get('TRIGGER_TRAIN_K_FOLDS_MIN_HOURS')
            if finished_ts is not None:
                trigger_min_ts = finished_ts.shift(hours=min_hours)
                if arrow.utcnow() < trigger_min_ts:
                    print('k-fold {}: exit on time constraint'.format(k_fold))
                    continue

        # train if no models exist
        if get_learned_models_count(problem_name, k_fold=k_fold) == 0:
            reason = 'first model'
            trigger_train = True

        # number of new samples (new annotated or added from script)
        n_new = 0
        if not trigger_train:
            n_samples = app.db.session.query(GenderSample). \
                filter(and_(GenderSample.is_hard == False,  # no bad or hard samples
                            GenderSample.is_bad == False,
                            GenderSample.always_test == False,
                            GenderSample.k_fold != None,
                            GenderSample.k_fold != k_fold)). \
                outerjoin(GenderUserAnnotation). \
                filter(or_(and_(GenderUserAnnotation.id == None,
                                GenderSample.is_annotated_gt),
                           and_(GenderUserAnnotation.id != None,
                                GenderUserAnnotation.is_hard == False,
                                GenderUserAnnotation.is_bad == False))).count()

            model = LearnedModel.query.filter(and_(LearnedModel.problem_name == problem_name,
                                                   LearnedModel.k_fold == k_fold,
                                                   LearnedModel.finished_ts != None)).first()
            if model:
                n_new = n_samples - model.num_samples
                min_samples = app.config.get('TRIGGER_TRAIN_K_FOLDS_MIN_SAMPLES')
                if n_new >= min_samples:
                    trigger_train = True
                    reason = '{} new samples'.format(n_new)

        # number of changed samples
        n_changed_samples = 0
        if not trigger_train:
            n_changed_samples = app.db.session.query(GenderSample). \
                filter(and_(GenderSample.always_test == False,
                            GenderSample.k_fold != None,
                            GenderSample.k_fold != k_fold,
                            GenderSample.is_checked == True,
                            GenderSample.is_changed == True)).count()

            min_samples = app.config.get('TRIGGER_TRAIN_K_FOLDS_MIN_SAMPLES')
            if n_changed_samples >= min_samples:
                trigger_train = True
                reason = '{} samples have been annotated'.format(n_changed_samples)

        if not trigger_train and (n_changed_samples > 0 or n_new > 0):
            finished_ts = get_finished_time_task(problem_name, problem_type, k_fold=k_fold)

            if finished_ts is not None:
                last_train_ts = finished_ts
                trigger_max_hours = app.config.get('TRIGGER_TRAIN_K_FOLDS_MAX_HOURS')
                trigger_train_ts = last_train_ts.shift(hours=trigger_max_hours)

                if arrow.utcnow() >= trigger_train_ts:
                    trigger_train = True
                    reason = 'time threshold'

        if trigger_train:
            print('run k-fold {} auto-training: {}'.format(k_fold, reason))
            clear_old_tasks(problem_name, problem_type, k_fold=k_fold)

            task_id = celery.uuid()
            utc = arrow.utcnow()
            task_db = LearningTask(problem_name=problem_name, problem_type=problem_type,
                                   task_id=task_id, started_ts=utc, k_fold=k_fold)

            app.db.session.add(task_db)
            app.db.session.flush()
            app.db.session.commit()

            task = run_train_k_folds.apply_async((problem_name, k_fold), task_id=task_id,
                                                 link_error=train_k_folds_on_error.s(),
                                                 link=train_k_folds_on_success.s(),
                                                 queue='learning')
            task_ids.append(task_id)
            folds_tested.append(k_fold)
            print('{} task successfully started'.format(task.id))

    if len(task_ids) > 0:
        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += 'run training for folds={}'.format(','.join([str(fold) for fold in folds_tested]))

        send_slack_message(msg)

@app.celery.task()
def auto_testing():
    problem_name = 'gender'
    problem_type = 'test'

    if check_working_tasks(problem_name, problem_type):
        print('auto-testing: testing already running, skip')
        return

    if get_learned_models_count(problem_name) == 0:
        print('auto-testing: no models to test')
        return

    do_run_test = False
    reason = ''

    # check number of models not tested
    if not do_run_test:
        n_not_tested_models = LearnedModel.query.filter(and_(LearnedModel.problem_name == problem_name,
                                                             LearnedModel.k_fold == None,
                                                             LearnedModel.finished_ts != None)).\
            outerjoin(AccuracyMetric).filter(AccuracyMetric.id==None).count()

        if n_not_tested_models > 0:
            reason = 'some models is not tested'
            do_run_test = True

    # check time
    if not do_run_test:
        finished_ts = get_finished_time_task(problem_name, problem_type)
        min_minutes = app.config.get('TRIGGER_TEST_MIN_MINUTES')
        if finished_ts is not None:
            trigger_min_ts = finished_ts.shift(minutes=min_minutes)
            if arrow.utcnow() < trigger_min_ts:
                print('auto-testing: exit on time constraint')
                return

    # find not tested samples
    if not do_run_test:
        n_not_tested_samples = app.db.session.query(GenderSample). \
            filter(GenderSample.always_test == True). \
            outerjoin(GenderSampleResult).filter(GenderSampleResult.id == None).count()
        if n_not_tested_samples > 0:
            reason = 'new samples have been added'
            do_run_test = True

    # find changed samples
    if not do_run_test:
        n_changed_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.always_test == True,
                        GenderSample.is_checked == True,
                        GenderSample.is_changed == True)).count()
        if n_changed_samples > 0:
            reason = '{} samples have been changed'.format(n_changed_samples)
            do_run_test = True

    # check number of not checked samples
    if not do_run_test:
        n_not_checked_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.always_test == True,
                        GenderSample.is_checked == False)).count()

        n_samples = app.db.session.query(GenderSample). \
            filter(GenderSample.always_test == True).count()

        if n_not_checked_samples == 0 and n_samples > 0:
            reason = 'all samples have been checked'
            do_run_test = True

    # run testing
    if do_run_test:
        print('run auto-testing: {}'.format(reason))

        clear_old_tasks(problem_name, problem_type)

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name, problem_type=problem_type,
                               task_id=task_id, started_ts=utc)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_test.apply_async((problem_name,), task_id=task_id,
                                    link_error=test_on_error.s(), link=test_on_success.s(),
                                    queue='learning')

        print('auto-testing: {} task successfully started'.format(task.id))

        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += '{}, run testing'.format(reason)

        send_slack_message(msg)
    else:
        print('do not run auto-testing: no changed samples, not all samples are checked')

@app.celery.task()
def auto_testing_k_folds():
    print('auto-testing k-folds')

    problem_name = 'gender'
    problem_type = 'test_k_folds'

    k_folds = app.config.get('CV_PARTITION_FOLDS')

    task_ids = []
    folds_tested = []
    for k_fold in range(k_folds):

        trigger_test = False
        reason = ''

        if check_working_tasks(problem_name, problem_type, k_fold=k_fold):
            print('k-fold {}: attempted to start testing while other task not finished'.format(k_fold))
            continue

        if get_learned_models_count(problem_name, k_fold=k_fold) == 0:
            print('k-fold {}: no models to test'.format(k_fold))
            continue

        # check time
        if not trigger_test:
            finished_ts = get_finished_time_task(problem_name, problem_type, k_fold=k_fold)
            min_minutes = app.config.get('TRIGGER_TEST_K_FOLDS_MIN_MINUTES')
            if finished_ts is not None:
                trigger_min_ts = finished_ts.shift(minutes=min_minutes)
                if arrow.utcnow() < trigger_min_ts:
                    print('k-fold {}: exit on time constraint'.format(k_fold))
                    continue

        # check not tested samples
        if not trigger_test:
            n_not_tested_samples = app.db.session.query(GenderSample). \
                filter(and_(GenderSample.always_test == False,
                            GenderSample.k_fold == k_fold)). \
                outerjoin(GenderSampleResult).filter(GenderSampleResult.id == None).count()
            if n_not_tested_samples > 0:
                reason = 'new samples have been added'
                trigger_test = True

        # check number of not checked samples
        if not trigger_test:
            n_not_checked_samples = app.db.session.query(GenderSample). \
                filter(and_(GenderSample.always_test == False,
                            GenderSample.k_fold == k_fold,
                            GenderSample.is_checked == False)).count()

            n_samples = app.db.session.query(GenderSample). \
                filter(and_(GenderSample.always_test == False,
                            GenderSample.k_fold == k_fold)).count()

            if n_not_checked_samples == 0 and n_samples > 0:
                reason = 'all samples have been checked'
                trigger_test = True

        if trigger_test:
            print('run k-fold {} auto-testing: {}'.format(k_fold, reason))

            clear_old_tasks(problem_name, problem_type, k_fold=k_fold)

            task_id = celery.uuid()
            utc = arrow.utcnow()
            task_db = LearningTask(problem_name=problem_name, problem_type=problem_type,
                                   task_id=task_id, started_ts=utc, k_fold=k_fold)
            app.db.session.add(task_db)
            app.db.session.flush()
            app.db.session.commit()

            task = run_test_k_folds.apply_async((problem_name, k_fold), task_id=task_id,
                                                link_error=test_k_folds_on_error.s(),
                                                link=test_k_folds_on_success.s(),
                                                queue='learning')
            task_ids.append(task_id)
            folds_tested.append(k_fold)
            print('{} task successfully started'.format(task.id))
        else:
            print('k-fold {}: do not run testing: no changed samples, not all samples are checked'.format(k_fold))

    if len(task_ids) > 0:
        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += 'run testing for folds={}'.format(','.join([str(fold) for fold in folds_tested]))

        send_slack_message(msg)

@app.celery.task()
def auto_deploy():

    problem_name = 'gender'
    problem_type = 'deploy'

    if check_working_tasks(problem_name, problem_type):
        print('auto-deploy: already running, skip')
        return

    if get_learned_models_count(problem_name) == 0:
        print('auto-deploy: no models')
        return

    top_model =  app.db.session.query(LearnedModel, AccuracyMetric).\
        filter(and_(LearnedModel.k_fold==None,
                    LearnedModel.prefix!=None,
                    LearnedModel.epoch!=None,
                    LearnedModel.problem_name==problem_name)).\
        join(AccuracyMetric).order_by(desc(AccuracyMetric.accuracy)).first()

    if top_model is None:
        print('auto-deploy: models is not tested')
        return

    if top_model.LearnedModel.is_deployed:
        print('auto-deploy: model is already deployed')
        return

    trigger_deploy = False
    reason = ''

    if not trigger_deploy:
        finished_ts = get_finished_time_task(problem_name, problem_type)
        min_hours = app.config.get('TRIGGER_DEPLOY_MIN_HOURS')
        if finished_ts is not None:
            trigger_min_ts = finished_ts.shift(hours=min_hours)
            if arrow.utcnow() < trigger_min_ts:
                print('auto-deploy: exit on time constraint')
                return

    if not trigger_deploy:
        min_accuracy = app.config.get('MIN_ACCURACY_TO_DEPLOY')
        if top_model.AccuracyMetric.accuracy < min_accuracy:
            print('auto-deploy: accuracy {} is not enough'.format(top_model.AccuracyMetric.accuracy))
            return

    trigger_deploy = True

    if trigger_deploy:
        print('auto-deploy: model #{}'.format(top_model.LearnedModel.id))

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name, problem_type=problem_type,
                               task_id=task_id, started_ts=utc)

        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_deploy.apply_async((problem_name, top_model.LearnedModel.id), task_id=task_id,
                                    link_error=deploy_on_error.s(), link=deploy_on_success.s(),
                                    queue='learning')

        print('auto-deploy: {} task successfully started'.format(task.id))

    return