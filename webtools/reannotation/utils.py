from webtools import app

from .models import Image, GenderSample, GenderUserAnnotation, LearningTask, LearnedModel,\
    AccuracyMetric, GenderSampleResult

from sqlalchemy import func, or_, and_, desc, not_

def clear_old_tasks(problem_name, problem_type):
    learning_task = LearningTask.query.filter_by(problem_name=problem_name,problem_type=problem_type)
    print('{} tasks successfully deleted'.format(learning_task.count()))
    learning_task.delete()
    app.db.session.flush()
    app.db.session.commit()

def check_working_tasks(problem_name, problem_type):
    return LearningTask.query.\
               filter_by(problem_type=problem_type,problem_name=problem_name,finished_ts=None)\
               .count() > 0

def get_learned_models_count(problem_name, k_folds=False):
    if not k_folds:
        models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                                LearnedModel.k_fold==None,
                                                LearnedModel.finished_ts!=None))
    else:
        models = LearnedModel.query.filter(and_(LearnedModel.problem_name==problem_name,
                                                LearnedModel.k_fold!=None,
                                                LearnedModel.finished_ts!=None))
    return models.count()