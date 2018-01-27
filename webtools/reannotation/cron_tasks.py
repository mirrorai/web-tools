from webtools import app
import arrow
from .models import GenderSample, GenderUserAnnotation, LearnedModel, LearningTask
from .models import AccuracyMetric, GenderSampleResult
from sqlalchemy import func, or_, and_, desc, not_
from webtools.utils import  send_slack_message
from .utils import clear_old_tasks, check_working_tasks, get_learned_models_count, get_finished_time_task
import celery
from .celery_tasks import test_on_error, test_on_success, run_test, run_train, train_on_error, train_on_success

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

    if check_working_tasks(problem_name, 'train'):
        print('auto-training: training already running, skip')
        return

    trigger_train = False
    reason = ''

    if get_learned_models_count(problem_name) == 0:
        reason = 'first model'
        trigger_train = True

    if not trigger_train:
        n_changed_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.always_test == False,
                        GenderSample.is_checked == True,
                        GenderSample.is_changed == True)).count()

        if n_changed_samples == 0:
            print('no changed samples')
            return

        min_samples = app.config.get('TRIGGER_TRAIN_MIN_SAMPLES')
        if n_changed_samples >= min_samples:
            trigger_train = True
            reason = '{} samples have been annotated'.format(n_changed_samples)

    if not trigger_train:
        finished_ts = get_finished_time_task(problem_name, 'train')

        if finished_ts is not None:
            last_train_ts = finished_ts
            trigger_max_hours = app.config.get('TRIGGER_TRAIN_MAX_HOURS')
            trigger_train_ts = last_train_ts.shift(hours=trigger_max_hours)

            if arrow.utcnow() >= trigger_train_ts:
                trigger_train = True
                reason = 'time threshold'

    if trigger_train:
        print('run auto-training')

        clear_old_tasks(problem_name, 'train')

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name, problem_type='train',
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


@app.celery.task()
def auto_testing():
    problem_name = 'gender'

    if check_working_tasks(problem_name, 'test'):
        print('auto-testing: testing already running, skip')
        return

    if get_learned_models_count(problem_name) == 0:
        print('auto-testing: no models to test')
        return

    do_run_test = False
    reason = ''

    n_not_tested_models = LearnedModel.query.filter(LearnedModel.problem_name == problem_name).\
        outerjoin(AccuracyMetric).filter(AccuracyMetric.id==None).count()
    if n_not_tested_models > 0:
        reason = 'some models is not tested'
        do_run_test = True

    if not do_run_test:
        finished_ts = get_finished_time_task(problem_name, 'test')
        min_minutes = app.config.get('TRIGGER_TEST_MIN_MINUTES')
        if finished_ts is not None:
            trigger_min_ts = finished_ts.shift(minutes=min_minutes)
            if arrow.utcnow() < trigger_min_ts:
                print('exit on time constraint')
                return

    if not do_run_test:
        n_not_tested_samples = app.db.session.query(GenderSample). \
            filter(GenderSample.always_test == True). \
            outerjoin(GenderSampleResult).filter(GenderSampleResult.id == None).count()
        if n_not_tested_samples > 0:
            reason = 'new samples have been added'
            do_run_test = True

    if not do_run_test:
        n_changed_samples = app.db.session.query(GenderSample). \
            filter(and_(GenderSample.always_test == True,
                        GenderSample.is_checked == True,
                        GenderSample.is_changed == True)).count()
        if n_changed_samples > 0:
            reason = '{} samples have been changed'.format(n_changed_samples)
            do_run_test = True

    if do_run_test:
        print('run auto-testing')

        clear_old_tasks('gender', 'test')

        task_id = celery.uuid()
        utc = arrow.utcnow()
        task_db = LearningTask(problem_name=problem_name, problem_type='test',
                               task_id=task_id, started_ts=utc)
        app.db.session.add(task_db)
        app.db.session.flush()
        app.db.session.commit()

        task = run_test.apply_async((problem_name,), task_id=task_id,
                                    link_error=test_on_error.s(), link=test_on_success.s(),
                                    queue='learning')

        print('{} task successfully started'.format(task.id))

        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += '{}, run testing'.format(reason)

        send_slack_message(msg)


@app.celery.task()
def auto_deploy():
    print('auto deploy')
    return