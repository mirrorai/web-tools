from webtools import app
import arrow
from .models import GenderSample, GenderUserAnnotation, LearnedModel, LearningTask
from .models import AccuracyMetric, GenderSampleResult
from sqlalchemy import func, or_, and_, desc, not_
from webtools.utils import  send_slack_message
from .utils import clear_old_tasks, check_working_tasks, get_learned_models_count
import celery
from .celery_tasks import test_on_error, test_on_success, run_test

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
    print('auto-training')
    pass

@app.celery.task()
def auto_testing():
    problem_name = 'gender'

    if check_working_tasks(problem_name, 'test'):
        print('auto-testing: testing already running, skip')
        return

    if get_learned_models_count(problem_name) == 0:
        print('auto-testing: no models to test')
        return None

    do_run_test = False
    reason = ''

    n_not_tested_models = LearnedModel.query.filter(LearnedModel.problem_name == problem_name).\
        outerjoin(AccuracyMetric).filter(AccuracyMetric.id==None).count()
    if n_not_tested_models > 0:
        reason = 'models is not tested'
        do_run_test = True

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
                        GenderSample.checked_times == 0)).count()
        if n_changed_samples > 0:
            reason = 'samples have been changed'
            do_run_test = True

    if do_run_test:
        print('run auto-testing')

        msg = ':vertical_traffic_light: Scheduled check\n*Gender*: '
        msg += '{}, run testing'.format(reason)

        send_slack_message(msg)

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


@app.celery.task()
def auto_deploy():
    print('auto deploy')
    return