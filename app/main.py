from flask import Flask
from flask_restx import Api, Resource, reqparse
import json
import os
from models import prepare_data, fitting, prediction

app = Flask(__name__)
api = Api(app)

# подстановочные поля для обучения модели
params_to_fit_model = reqparse.RequestParser()
params_to_fit_model.add_argument('id_model', help='Уникальный ID модели, пример: 13', required=True)
params_to_fit_model.add_argument('name_model',  help='Название модели для обучения', choices=['RandomForestRegressor', 'LinearRegression'], required=True)
params_to_fit_model.add_argument('model_params', help='Параметры для обучения модели, пример {"n_estimators": 100}')

# подстановочные поля для предсказания модели
params_to_predict = reqparse.RequestParser()
params_to_predict.add_argument('id_model', help='ID модели для предсказания', required=True)

# подстановочные поля для удаления обученной модели
params_to_delete_model = reqparse.RequestParser()
params_to_delete_model.add_argument('id_model', help='ID модели для удаления', required=True)


@api.route('/all_available_models', methods=['GET'], doc={'description': 'Получить названия доступных моделей для обучения'})
class All_Available_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        return 'Доступные модели регрессии для обучения: RandomForestRegressor, LinearRegression', 200


@api.route('/all_trained_models', methods=['GET'], doc={'description': 'Получить информацию об имеющихся обученных моделях'})
class All_Trained_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        with open("param_fitted_models.json", "r") as jsonFile:
            trained_models = json.load(jsonFile)
        if trained_models == {}:
            return 'Нет доступных обученных моделей', 200
        else:
            return trained_models, 200


@api.route('/fit_model', methods=['POST'], doc={'description': 'Обучить и сохранить модель с выбранными параметрами'})
class Fit_Model(Resource):
    @api.expect(params_to_fit_model)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def post(self):
        args = params_to_fit_model.parse_args()
        id_model = args.id_model
        name_model = args.name_model
        model_params = {} if args.model_params is None else json.loads(args.model_params.replace("'", "\""))
        if id_model + '.pkl' in os.listdir('fitted_models/'):
            return 'Модель с таким ID уже существует, выберите другое ID', 400
        else:
            x, y = prepare_data('car_price_prediction.csv')
            status, param = fitting(x, y, name_model, id_model, model_params)
            if status == 'error':
                return f'Параметр {param} у модели {name_model} не найден', 400
            else:
                # сохраняю все параметры обученной модели
                with open("param_fitted_models.json", "r") as jsonFile:
                    data = json.load(jsonFile)
                data[id_model] = {'name_model': name_model, 'model_params': model_params}
                with open("param_fitted_models.json", "w") as jsonFile:
                    json.dump(data, jsonFile)
                return 'Модель успешно обучена', 200


@api.route('/predict', methods=['GET'], doc={'description': 'Сделать предсказание с помощью выбранной обученной модели'})
class Predict(Resource):
    @api.expect(params_to_predict)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        args = params_to_predict.parse_args()
        id_model = args.id_model
        if id_model + '.pkl' not in os.listdir('fitted_models/'):
            return f'Обученной модели с ID = {id_model} не существует', 400
        else:
            x, y = prepare_data('car_price_prediction.csv')
            y_pred = prediction(id_model, x)
            return json.dumps({f'ID = {id_model}': y_pred.astype(int).tolist()})


@api.route('/delete_model')
class Delete_Model(Resource):
    @api.expect(params_to_delete_model)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def delete(self):
        args = params_to_delete_model.parse_args()
        id_model = args.id_model
        if id_model + '.pkl' not in os.listdir('fitted_models/'):
            return f'Обученной модели с ID = {id_model} не существует', 400
        else:
            with open("param_fitted_models.json", "r") as file:
                data = json.load(file)
            del data[id_model]
            with open("param_fitted_models.json", "w") as jsonFile:
                json.dump(data, jsonFile)
            os.remove('fitted_models/' + str(id_model) + '.pkl')
            return f'Модель с ID = {id_model} удалена', 200


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port="5000")


