from webtools import app

from .models import Image, GenderSample, GenderUserAnnotation, LearningTask, LearnedModel,\
    AccuracyMetric, GenderSampleResult

from sqlalchemy import func, or_, and_, desc, not_

def clear_old_tasks(problem_name, problem_type, k_fold=None):
    learning_task = LearningTask.query.filter_by(problem_name=problem_name,k_fold=k_fold,problem_type=problem_type)
    print('{} tasks successfully deleted'.format(learning_task.count()))
    learning_task.delete()
    app.db.session.flush()
    app.db.session.commit()

def check_working_tasks(problem_name, problem_type, k_fold=None):
    return LearningTask.query.\
               filter_by(problem_type=problem_type,k_fold=k_fold,problem_name=problem_name,finished_ts=None)\
               .count() > 0

def get_learned_models_count(problem_name, k_fold=None):
    models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                            LearnedModel.k_fold==k_fold,
                                            LearnedModel.finished_ts!=None))
    return models.count()

def get_all_k_folds_learned_models_count(problem_name):
    models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                            LearnedModel.k_fold!=None,
                                            LearnedModel.finished_ts!=None))
    return models.count()

def get_finished_time_task(problem_name, problem_type, k_fold=None):
    task =  LearningTask.query. \
               filter_by(problem_type=problem_type, k_fold=k_fold, problem_name=problem_name) \
               .first()
    if task is None:
        return None
    elif task.finished_ts is None:
        return None
    else:
        return task.finished_ts