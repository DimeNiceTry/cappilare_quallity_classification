"""Build and execute the focused Russian-language notebook collection."""
import argparse
import asyncio
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


SETUP = """from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'focus_classifier' / 'features.py').is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fast_focus_4.workbench import (
    HERE, ART, PICTURES, BASIC, MODEL_IDS, load_data, baseline_table,
    load_model, predict_files, benchmark, run_ablations, photo_grid,
)
from focus_classifier.report import decision_metrics, probability_metrics
from research.evaluate_calibration import calibration_bins
plt.rcParams.update({'figure.dpi': 110, 'font.size': 10})
table = load_data()
valid = table.loc[table.y >= 0].copy()
print(f'Корень: {ROOT}')
print(f'Всего {len(table)}; для обучения/метрик {len(valid)}; '
      f'ok={int(valid.y.sum())}, not_ok={int((valid.y == 0).sum())}; '
      f'пациентов={valid.patient_id.nunique()}')
"""


def baseline_notebook():
    return [
        md("""
        # 01. Четыре признака: как устроен классификатор и где он ошибается

        Основная модель — **логистическая регрессия `logistic_basic`**.
        ExtraTrees на тех же четырёх признаках оставлен для сравнения.
        Здесь загружаются существующие результаты: обучение не запускается.

        Графики и фотографии сохранены в ноутбуке. Для повторного расчёта графиков
        достаточно файлов репозитория; оригинальные фотографии нужны только для галереи.
        Порядок работы и установка — в [README.md](README.md).
        """),
        code(SETUP),
        md("""
        ## Что означает метка

        `ok=1`: границы мелких и крупных сосудов основной сети различимы по полю кадра.
        Размытый глубокий фон допустим, если основная сеть чёткая.
        `not_ok=0`: кадр этим требованиям не соответствует.
        Неуверенность и расхождения **не превращаем в плохие кадры**: 435 изображений
        исключены из бинарной оценки. У оставшихся 565 кадров согласованы историческая
        и перекрёстная оценки; согласие не гарантирует истинность разметки.

        Все оценки ниже — OOF: пациент фотографии отсутствовал при обучении соответствующей
        внешней модели. Это пять частей по пациентам, с тремя внутренними частями
        для калибровки и порогов. Выбор между моделями остаётся исследовательским.
        """),
        code("""
        display(pd.crosstab(valid['fold'], valid['y']).rename(columns={0:'not_ok', 1:'ok'}))
        report = json.loads((ART / 'metrics.json').read_text(encoding='utf-8'))
        display(baseline_table().round(4))
        """),
        md("""
        ## Какие именно четыре признака

        Вход — 8-битный кадр **1024×768**, перевод в серый и деление на 255.
        `a` — нормированное изображение, `s` — GaussianBlur с σ=1 пиксель.

        | Признак | Расчёт | Ограничение |
        | --- | --- | --- |
        | `laplacian_raw` | Дисперсия Laplacian(a), ядро 1 | Шум тоже увеличивает значение |
        | `laplacian_smooth` | Дисперсия Laplacian(s), ядро 1 | Сглаживание ослабляет мелкие детали |
        | `tenengrad_smooth` | Среднее SobelX(s)² + SobelY(s)², ядра 3×3 | Зависит от контраста и количества границ |
        | `local_tenengrad_p75` | p75 средних энергий градиента по сетке 4×4 | Чёткая часть может скрыть размытие остального поля |

        Сетка нужна только для одного статистического признака: отдельные классификаторы
        частей изображения не обучаются. Реализация общая с рабочим клиентом:
        `focus_classifier/features.py`, режим `basic_only=True`.
        """),
        code("""
        from sklearn.metrics import roc_auc_score
        display(pd.DataFrame([dict(feature=f, auroc_higher_is_better=roc_auc_score(valid.y, valid[f]))
                              for f in BASIC]).round(4))
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
        for ax, feature in zip(axes.flat, BASIC):
            for label, color in [(0, 'tab:red'), (1, 'tab:green')]:
                ax.hist(np.log10(np.maximum(valid.loc[valid.y == label, feature], 1e-10)),
                        bins=25, density=True, alpha=.5, label=['not_ok', 'ok'][label], color=color)
            ax.set(title=feature, xlabel='log10(feature)', ylabel='Density')
            ax.legend()
        (HERE/'figures').mkdir(exist_ok=True)
        fig.savefig(HERE/'figures'/'feature_distributions.png', dpi=140)
        plt.show()
        """),
        md("""
        Эти AUROC относятся к **сырым значениям на 565 согласованных кадрах**:
        направление «больше = лучше» задано заранее. Они не равны AUROC обученной
        одномерной модели и не сравниваются напрямую со старой оценкой на 790 метках.

        ## Как четыре числа становятся вероятностью

        Для LR: `log10(max(feature, 1e-10))` → StandardScaler → LogisticRegression
        (`C=1`, `class_weight='balanced'`) → sigmoid-калибровка исходной вероятности.
        Масштабы обучаются внутри соответствующей обучающей части.
        Итоговый рабочий экземпляр обучен на всех 565 кадрах; его коэффициенты ниже
        служат объяснением модели, а не независимой проверкой качества.
        """),
        code("""
        bundle = load_model('logistic_basic')
        pipeline = bundle['model']
        coefficients = pd.DataFrame({'feature': BASIC,
                                     'weight_after_log_and_standardization': pipeline[-1].coef_[0]})
        display(coefficients)
        print('Пороги итоговой модели:', bundle['thresholds'])
        print('Sigmoid-калибровка: slope=', bundle['calibrator'].coef_[0,0],
              'intercept=', bundle['calibrator'].intercept_[0])
        """),
        md("""
        Коэффициенты работают совместно: признаки коррелируют, поэтому знак коэффициента
        не доказывает, что конкретный физический эффект улучшает или ухудшает изображение.

        ## Ранжирование и решения

        AUROC — качество порядка изображений, AP — поиск редких `ok`.
        Precision — доля хороших среди принятых, recall — доля сохранённых хороших.
        На этом наборе всегда отклонять всё даёт 89% accuracy и нулевой recall,
        поэтому одной accuracy недостаточно.
        """),
        code("""
        from sklearn.metrics import roc_curve, precision_recall_curve
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for model in MODEL_IDS:
            p = valid[f'{model}__p'].to_numpy()
            fpr, tpr, _ = roc_curve(valid.y, p)
            precision, recall, _ = precision_recall_curve(valid.y, p)
            axes[0].plot(fpr, tpr, label=model)
            axes[1].plot(recall, precision, label=model)
        axes[0].plot([0,1], [0,1], 'k--', alpha=.4)
        axes[1].axhline(valid.y.mean(), ls='--', color='gray', label='OK prevalence')
        axes[0].set(xlabel='False accept rate', ylabel='Recall OK', title='OOF ROC')
        axes[1].set(xlabel='Recall OK', ylabel='Precision OK', title='OOF precision-recall')
        for ax in axes:
            ax.legend(); ax.grid(alpha=.2)
        fig.savefig(HERE/'figures'/'roc_pr.png', dpi=140)
        plt.show()
        display(pd.DataFrame([dict(model=m, **decision_metrics(valid.y.to_numpy(), valid[f'{m}__p'].to_numpy(),
                                      valid[f'{m}__balanced'].to_numpy())) for m in MODEL_IDS]).round(4))
        """),
        md("""
        В обычном режиме LR принимает **53 хороших и 106 плохих** фотографий,
        теряя 9 хороших. Это полезная точка отсчёта для улучшения, но недостаточная
        чистота автоматического отбора. Обычный порог максимизирует balanced accuracy
        на внутренних данных и не обязан равняться 0,5.

        Строгий порог требовал наблюдаемую precision ≥90% минимум на пяти принятых
        внутренних кадрах. Обе модели на четырёх признаках такой режим не обеспечили:
        служебный порог 2 означает **«автодопуск отключён»**, а не плохое качество кадра.

        ## ECE: соответствуют ли вероятности частотам

        `ECE_ok = Σ (n_bin/N) · |mean(P(ok)) − fraction(ok)|`.
        Ниже — десять равных интервалов, исходные и откалиброванные вероятности.
        Для сравнения выводим также top-label ECE: уверенность в выбранном при 0,5
        классе против факта правильности этого класса. Это другая величина, её
        нельзя незаметно смешивать с ECE вероятности `ok`.
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        ece_rows, bin_rows = [], []
        y = valid.y.to_numpy()
        for model in MODEL_IDS:
            for j, suffix in enumerate(['raw', 'p']):
                p = valid[f'{model}__{suffix}'].to_numpy()
                metrics = probability_metrics(y, p)
                ece, bins = calibration_bins(p, y, 10)
                occupied = [b for b in bins if b['count']]
                axes[j].plot([b['mean_probability'] for b in occupied],
                             [b['observed_frequency'] for b in occupied], 'o-', label=f'{model}, ECE={ece:.3f}')
                ece_rows.append(dict(model=model, variant=suffix,
                                     **{f'ECE_ok_{n}': metrics['ece_ok'][str(n)] for n in [5,10,15]},
                                     ECE_top_10=metrics['ece_top']['10'], Brier=metrics['brier']))
                bin_rows.extend(dict(model=model, variant=suffix, **b) for b in bins)
        for j, ax in enumerate(axes):
            ax.plot([0,1], [0,1], 'k--', alpha=.5)
            ax.set(xlim=(0,1), ylim=(0,1), xlabel='Mean P(ok)', ylabel='Observed OK fraction',
                   title=['Raw probabilities', 'Calibrated probabilities'][j])
            ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.savefig(HERE/'figures'/'calibration.png', dpi=140)
        plt.show()
        display(pd.DataFrame(ece_rows).round(4))
        display(pd.DataFrame(bin_rows).query("model == 'logistic_basic' and variant == 'p'"))
        """),
        md("""
        У LR ECE_ok(10) снижается примерно с 0,211 до 0,041.
        Малый ECE не означает качественный отбор: константный прогноз частоты класса
        тоже может иметь низкий ECE. В редких высоковероятностных интервалах мало
        фотографий; смотрите числа кадров в таблице, Brier, PR и реальные ошибки.

        ## Фотографии: правильные решения и уверенные ошибки

        Автоматически выбираем по два примера TP, FP, FN и TN по OOF-оценкам LR.
        Это иллюстрации, выбранные по результату проверки; они не являются отдельным тестом.
        Исходный масштаб фотографии важен: в превью мелкие границы могут потеряться.
        """),
        code("""
        fig = photo_grid(table, per_case=2)
        fig.savefig(HERE/'figures'/'photo_examples.png', dpi=115)
        plt.show()
        mistakes = valid.loc[((valid.y == 0) & (valid.logistic_basic__p >= valid.logistic_basic__balanced)) |
                             ((valid.y == 1) & (valid.logistic_basic__p < valid.logistic_basic__balanced))]
        display(mistakes.sort_values('logistic_basic__p', ascending=False)[
            ['file','patient_id','y','logistic_basic__p', *BASIC, 'review_comment']].head(12))
        """),
        md("""
        ## Что улучшать и как проверять

        Сначала отделить вклад признаков от выбора порога и ошибок разметки.
        Во втором ноутбуке воспроизводим LR-4 и исключаем по одному признаку,
        сохраняя одинаковые внешние части по пациентам. В третьем — оценка своих
        файлов и измерение времени. Подробные ограничения и план — в `METHOD_RU.md`.
        """),
    ]


def experiments_notebook():
    return [
        md("""
        # 02. Воспроизведение LR-4 и проверка вклада каждого признака

        Выполненный небольшой эксперимент: исходная модель и четыре варианта
        с исключением одного признака. Используем готовые значения признаков:
        исходные PNG не нужны, извлечение 43 признаков не запускается.

        Это проверка, какие из четырёх входов помогают текущей модели. Улучшения
        здесь исследовательские: на этих данных уже выбирались методы, поэтому
        победителя понадобится проверить на новой съёмке и новых пациентах.
        """),
        code(SETUP),
        md("""
        ## Зафиксированный протокол

        565 согласованных кадров: 62 `ok`, 503 `not_ok`. Пять внешних частей по
        пациентам, seed=20261102. В каждом внешнем обучении — три внутренние части
        для sigmoid-калибровки и порогов. Внешние метки используются только для
        расчёта метрик; StandardScaler обучается внутри train.

        Все варианты используют один алгоритм LR с C=1 и balanced-весами.
        Мы не выбираем порог на объединённых внешних прогнозах. Порог обычного
        режима и строгий порог определяются отдельно внутри каждого обучения.
        Воспроизведение исходной модели проверяется по всем вероятностям и порогам.
        """),
        code("""
        # True: пересчитать все пять вариантов. False: прочитать сохранённый результат.
        RUN_TRAINING = True
        OUT = HERE / 'results'
        if RUN_TRAINING:
            scores, predictions, folds = run_ablations(table, output_dir=OUT)
        else:
            scores = pd.read_csv(OUT/'ablation_metrics.csv')
            predictions = pd.read_csv(OUT/'ablation_oof.csv')
            folds = pd.read_csv(OUT/'ablation_folds.csv')
        display(scores[['experiment','n_features','auroc','auroc_delta_vs_baseline','ap',
                        'precision_ok','recall_ok','tp','fp','fn','ece_ok_10','strict_accepted']].round(4))
        print('Результаты сохранены только в', OUT)
        """),
        md("""
        `without_...` означает, что остальные три признака сохранены.
        Падение качества после удаления показывает полезность признака для этой
        модели; отсутствие падения может объясняться его избыточностью.
        Это не причинное объяснение физики резкости.

        ## Сравнение на одинаковых внешних пациентах

        Отдельные точки на графике — пять внешних частей. Их мало, размеры и
        количество хороших кадров различаются. Разброс не является 95% интервалом.
        Время обучения в CSV не является временем обработки одного изображения.
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
        labels = [s.replace('without_', 'minus\\n') for s in scores.experiment]
        axes[0].barh(labels, scores.auroc, color='steelblue')
        for i, name in enumerate(scores.experiment):
            v = folds.loc[folds.experiment == name, 'auroc']
            axes[0].scatter(v, np.full(len(v), i), c='black', s=16, zorder=3)
        axes[0].set(xlim=(0,1), xlabel='AUROC', title='OOF AUROC + outer-fold values')
        pos = np.arange(len(scores))
        axes[1].barh(pos-.18, scores.tp, height=.35, label='Good accepted', color='seagreen')
        axes[1].barh(pos+.18, scores.fp, height=.35, label='Bad accepted', color='indianred')
        axes[1].set(yticks=pos, yticklabels=labels, xlabel='Images', title='Inner-selected balanced thresholds')
        axes[1].legend()
        (HERE/'figures').mkdir(exist_ok=True)
        fig.savefig(HERE/'figures'/'ablation.png', dpi=140)
        plt.show()
        """),
        code("""
        # Парные разности AUROC: 1000 повторных выборок пациентов, фиксированные OOF-прогнозы.
        # Это исследовательские интервалы: не учитывают переобучение и множественный выбор вариантов.
        from sklearn.metrics import roc_auc_score
        wide = predictions.pivot(index='file', columns='experiment', values='p_ok').loc[valid.file]
        y = valid.y.to_numpy()
        groups = valid.patient_id.to_numpy()
        unique = np.unique(groups)
        patient_rows = {g: np.flatnonzero(groups == g) for g in unique}
        rng = np.random.default_rng(20261102)
        deltas = {n: [] for n in scores.experiment if n != 'baseline_4'}
        for _ in range(1000):
            ids = np.concatenate([patient_rows[g] for g in rng.choice(unique, len(unique), replace=True)])
            if len(np.unique(y[ids])) < 2:
                continue
            reference = roc_auc_score(y[ids], wide['baseline_4'].to_numpy()[ids])
            for name in deltas:
                deltas[name].append(roc_auc_score(y[ids], wide[name].to_numpy()[ids]) - reference)
        intervals = pd.DataFrame([dict(experiment=n, delta_ci95_low=np.percentile(v,2.5),
                                      delta_ci95_high=np.percentile(v,97.5)) for n,v in deltas.items()])
        display(intervals.round(4))
        intervals.to_csv(OUT/'paired_auc_delta_ci.csv', index=False, encoding='utf-8-sig')
        """),
        md("""
        ## Вывод из выполненного сравнения

        Исходные AUROC 0,87998 и решения 53 TP / 106 FP воспроизвелись.
        Удаление исходного лапласиана снижает AUROC до 0,87026: несмотря на слабое
        самостоятельное ранжирование, совместно с другими входами он полезен.
        Удаление сглаженного лапласиана меняет AUROC до 0,88084 — прирост меньше 0,001.
        Удаление локального p75 даёт AUROC 0,88142 и 56 TP / 105 FP;
        интервал разности AUROC включает ноль. Явного решения проблемы отбора нет:
        плохих допусков по-прежнему около сотни, строгий допуск отключён у всех вариантов.

        Поэтому четыре признака сохраняем как исходный вариант. Небольшие изменения
        не объявляем готовым улучшенным классификатором и не подменяем ими рабочие веса.
        """),
        md("""
        ## Как продолжать

        1. Пересмотреть уверенные ложные допуски и потери хороших из первого ноутбука.
        2. Заранее выбрать цель: например, повысить precision при сохранении нужного recall
           и не увеличить p95 времени обработки. Числа требований нужно согласовать с задачей.
        3. Менять по одному фактору: модель над четырьмя числами, обработка шума,
           нормировка контраста, затем масштаб изображения. Изменение вычисления признаков
           требует заново пересчитать признаки **и обучить модель**, старые веса не подходят.
        4. Для подбора C, полиномиальных взаимодействий или параметров дерева выбирать их
           только внутри внешнего train с группировкой пациентов. Не выбирать по этому
           OOF лучшие параметры и не называть ту же оценку независимым тестом.
        5. После фиксации метода проверить новую съёмку и новую разметку; отдельно измерить
           задержку на целевом компьютере. Спорные кадры не использовать как `not_ok`.

        Уменьшение числа входов не гарантирует ускорения извлечения: текущая функция
        `basic_only=True` всё равно считает все четыре. Здесь измерено влияние на качество.
        Модели клиента и исходные артефакты этим ноутбуком не заменяются.
        """),
    ]


def inference_notebook():
    return [
        md("""
        # 03. Свои фотографии и время обработки LR-4

        Используется сохранённая итоговая модель, обученная на всех 565 согласованных кадрах.
        Код работает без PyTorch и GPU. Время загрузки модели исключено из бенчмарка.
        Для независимой проверки нужны новые файлы и их разметка; изображения исходного
        набора здесь служат только демонстрацией работы.
        """),
        code(SETUP),
        md("""
        ## Указать файл или папку

        В `INPUT_PATH` можно записать путь к одной фотографии или папке с PNG/JPEG/BMP/TIFF.
        Пустой путь выбирает несколько примеров из исходного набора, если он скачан.
        Размер должен быть **1024×768**, 8 бит: автоматическое изменение масштаба
        меняет признаки, поэтому здесь оно не выполняется.
        """),
        code("""
        INPUT_PATH = ''  # Например: r'C:\\data\\my_capillary_photos'
        MODEL = 'logistic_basic'  # Другой вариант: 'trees_basic'
        MODE = 'balanced'        # 'precise' у LR-4 отключает автодопуск
        MANUAL_THRESHOLD = None  # Можно задать число от 0 до 1 для просмотра, не для оценки качества
        EXPORT_CSV = False

        if INPUT_PATH:
            source = Path(INPUT_PATH).expanduser()
            if not source.exists():
                raise FileNotFoundError(source)
            files = ([source] if source.is_file() else
                     sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}))
            if not files:
                raise ValueError('В папке нет поддерживаемых фотографий')
        else:
            demo = pd.concat([valid.loc[valid.y == 1].head(3), valid.loc[valid.y == 0].head(3)])
            files = [PICTURES/name for name in demo.file if (PICTURES/name).is_file()]
        if files:
            output = predict_files(files, model=MODEL, mode=MODE, threshold=MANUAL_THRESHOLD)
            display(output.round(5))
            if output.known_training_image.any():
                print('Есть фотографии обучения: показанные прогнозы не являются независимой оценкой.')
            if EXPORT_CSV:
                (HERE/'local_outputs').mkdir(exist_ok=True)
                output.to_csv(HERE/'local_outputs'/'my_predictions.csv', index=False, encoding='utf-8-sig')
        else:
            print('Демо-фотографии не скачаны. Укажи INPUT_PATH или выполни download_data.cmd в корне.')
        """),
        md("""
        `p_ok` — откалиброванная оценка вероятности `ok` в рамках обученной задачи.
        Порог не обязательно 0,5. `auto_accept_disabled` означает отсутствие подходящего
        строгого порога; это **не метка плохой фотографии**.
        Ручной порог помогает посмотреть поведение, но выбирать его по новым проверочным
        меткам и сообщать затем качество на тех же метках нельзя.

        ## Посмотреть входные фотографии

        Это уменьшенные превью. Оригиналы нужны для визуальной проверки границ мелких сосудов.
        """),
        code("""
        if files:
            from PIL import Image
            chosen = files[:6]
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
            for ax in axes.flat:
                ax.axis('off')
            for ax, file in zip(axes.flat, chosen):
                with Image.open(file) as image:
                    ax.imshow(image)
                row = output.loc[output.file == str(file)].iloc[0]
                ax.set_title(f'{file.name}\\nP(ok)={row.p_ok:.3f}; {row.decision}', fontsize=8)
            plt.show()
        """),
        md("""
        ## Замер скорости на этом компьютере

        Бенчмарк отдельно измеряет чтение/декодирование, четыре признака и классификатор.
        Один поток OpenCV, последовательная обработка, batch=1; первые три вызова
        прогревочные. Ни хеширование для поиска известных кадров, ни загрузка модели,
        ни графики в это время не входят. Файловый кэш ОС может быть прогрет.
        Несколько фотографий дают только грубую оценку p95; для устойчивого замера
        укажи папку и увеличь `N_BENCHMARK`.
        """),
        code("""
        import platform
        N_BENCHMARK = 30
        benchmark_files = (sorted(PICTURES.glob('*.png'))[:N_BENCHMARK] if not INPUT_PATH
                           else files[:N_BENCHMARK])
        if benchmark_files:
            timings = benchmark(benchmark_files, model=MODEL, warmup=3)
            display(timings.drop(columns='file').agg(['median', lambda x: x.quantile(.95)]).rename(index={'<lambda>':'p95'}).round(3))
            print('Фотографий:', len(timings), 'Последовательно, кадров/с:', round(1000/timings.total_ms.mean(), 2))
            print('CPU:', platform.processor(), 'Python:', platform.python_version())
            fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
            names = ['read_decode_ms','features_ms','head_ms','total_ms']
            ax.bar(names, timings[names].median(), label='median')
            ax.scatter(names, timings[names].quantile(.95), color='black', label='p95', zorder=3)
            ax.set(ylabel='Milliseconds', title=f'Current machine; {MODEL}; n={len(timings)}')
            ax.legend(); plt.show()
        else:
            print('Для замера нужны фотографии.')
        """),
        md("""
        В исходном замере на i5-9300H у LR-4: PNG → решение p50≈63,5 мс,
        p95≈140,8 мс; анализ уже декодированного массива p50≈31,3 мс.
        Это измерение конкретной реализации и компьютера, а не гарантия частоты видеопотока.
        Нельзя складывать медианы отдельных этапов и считать это медианой общей задержки.

        Для исходных кадров качество оцениваем по OOF из первого ноутбука;
        итоговые прогнозы этого ноутбука показывают работу рабочего экземпляра.
        """),
    ]


def main():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-execute', action='store_true')
    parser.add_argument('--only', choices=['01', '02', '03'])
    args = parser.parse_args()
    notebooks = [('01_baseline_and_errors_RU', baseline_notebook),
                 ('02_reproduce_and_ablate_RU', experiments_notebook),
                 ('03_predict_and_speed_RU', inference_notebook)]
    for name, build in notebooks:
        if args.only and not name.startswith(args.only):
            continue
        nb = nbf.v4.new_notebook(cells=build(), metadata={
            'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'},
            'language_info': {'name':'python', 'version':'3.10'},
        })
        if not args.no_execute:
            print('Executing', name, flush=True)
            NotebookClient(nb, timeout=600, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        nbf.validate(nb)
        path = HERE/f'{name}.ipynb'
        nbf.write(nb, path)
        print('Saved', path.name, flush=True)


if __name__ == '__main__':
    main()
