import Metrics
import ML
import Database

if __name__ == '__main__':

    database = Database.Database()
    model1 = ML.Model(database)
    print("модель создана:", model1.get_model())
    Metrics.get_cross_validation_score(model1)


