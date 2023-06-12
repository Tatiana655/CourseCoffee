import os
import dearpygui.dearpygui as dpg

import ML
import Database
import numpy as np


def get_samples(path_files, quality):
    all_data = []
    ans = []
    for filename in os.listdir(path_files):
        all_data.append(np.loadtxt(path_files + filename))
        ans.append(quality)
    return [all_data, ans]


def analysis(path, retry=False):
    data = np.loadtxt(path)
    if retry:
        dpg.add_text(f'Перезапуск {os.path.split(path)[-1]}!', parent="activity")
    else:
        dpg.add_text(f'{os.path.split(path)[-1]} успешно загружен!', parent="activity")
    if np.shape(data) == (300, 8):
        database = Database.Database()
        model1 = ML.Model(database)
        model1.get_model()

        try:
            dpg.add_text(f'Анализируем...', parent="activity")
            prediction = model1.get_proba_prediction(data)
            quality = np.argmax(prediction)

            if quality == 0:
                dpg.add_text(f'Качество кофе - ВЫСОКОЕ!', parent="activity", color=(54, 238, 24))
            elif quality == 1:
                dpg.add_text(f'Качество кофе - СРЕДНЕЕ!', parent="activity", color=(238, 217, 24))
            else:
                dpg.add_text(f'Качество кофе - НИЗКОЕ!', parent="activity", color=(238, 191, 24))
        except:
            dpg.add_text(f'[!] Ошибка анализа', parent="activity", color=(238, 24, 24))
            dpg.add_button(label="повторить", callback=callback, user_data=path, parent="activity")
            dpg.add_text(f'Попробуйте повторить операцию', parent="errors")
    else:
        dpg.add_text(f'[!] Ошибка чтения файла', parent="activity", color=(238, 24, 24))
        dpg.add_button(label="повторить", callback=callback, user_data=path, parent="activity")
        dpg.add_text(f'Поступили некорректные данные '
                     f'\n Ожидалось shape(300, 8);'
                     f'\n Получено shape{np.shape(data)}', parent="errors")
    dpg.add_separator(parent='activity')


def callback(sender, app_data, user_data):
    if user_data is not None:
        analysis(user_data, True)
    else:
        for item in app_data['selections']:
            analysis(app_data['selections'][item])
            dpg.add_text(f'Выбран {item}', parent="load")


def cancel_callback(sender, app_data):
    print('Cancel was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)


if __name__ == '__main__':
    dpg.create_context()

    # подгружаем кириллицу и устанавливаем шрифт по умолчанию
    with dpg.font_registry():
        with dpg.font("src/Inter-VariableFont_slnt,wght.ttf", 17, default_font=True, id="Default font"):
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    dpg.bind_font("Default font")

    # создаем окно выбора файла
    with dpg.file_dialog(directory_selector=False, show=False, callback=callback,
                         id="file_dialog_id", width=620, height=350):
        dpg.add_file_extension("", color=(217, 217, 217, 255))
        dpg.add_file_extension(".txt", color=(150, 255, 150, 255))

    # окно загрузки данных с носа
    with dpg.window(label="Загрузка данных с носа", tag='load', width=260, height=220):
        dpg.add_button(label="Выбрать файл", callback=lambda: dpg.show_item("file_dialog_id"),
                       width=220, height=44)

    # окно активности
    with dpg.window(label="Активность", tag='activity', width=440, height=440, pos=[260, 0]):
        pass

    # окно отчета по ошибкам
    with dpg.window(label="Отчет по ошибкам", tag='errors', width=260, height=220, pos=[0, 220]):
        pass

    dpg.create_viewport(title='Coffee analysis', width=700, height=440)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    # dpg.set_primary_window("Primary Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()
