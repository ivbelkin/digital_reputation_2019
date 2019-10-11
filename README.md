# Digital Reputation Challenge

Данное соревнование было выбрано в качестве домашней работы по
курсу "Методы искусственного интеллекта в анализе данных",
читаемого в МФТИ на магистерской программе "Методы и 
технологии искусственного интеллекта"

К сожалению, дедлайн соревнования был раньше, чем ДЗ, поэтому на лидерборде
есть только резльтаты SVM.

Лидерборд:
https://boosters.pro/championship/digital_reputation_challenge/rating

Команда: `cdsteam`

<table>
<caption>Rating</caption>
<thead><td>Place</td><td>Team</td><td>Solutions</td><td>Score</td></thead>
<tr><td>1</td><td>Mamat Shamshiev MMP MSU</td><td>28</td><td>0.6264040</td></tr>
<tr><td>2</td><td>Artem Tsypin</td><td>38</td><td>0.6260790</td></tr>
<tr><td>3</td><td>Polosataya</td><td>53</td><td>0.6195900</td></tr>
<tr><td>4-66</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>67</td><td>cdsteam</td><td>5</td><td>0.5853930</td></tr>
<tr><td>68-211</td><td>...</td><td>...</td><td>...</td></tr>
</table>

В целом для одиночного SVM не плохой результат, бейзлай на RandomForest, предоставленный организаторами
дает качество - 0.565919

PS: Медальки давали с 63 места :(

## Структура репозитория

`config.py` - параметры запуска

`dataset.py` - загрузка и предобработка данных

`train_svm.py` - поиск по сетке

`test_svm.py` - прогон моделей по результатам поиска по сетке на тестовых данных

`notebooks/EDA.ipynb` - разведывательный анализ данных

`notebooks/SVM.ipynb` - использовался исключительно для отладки

`submissions/` - результаты на тестовых данных

`output/screenfetch.txt` - характеристики железа

`ouput/svm_models.pkl` - обученные модели + параметры для DRCDataset

`output/progress_*` - вывод процесса обучения

`output/result_*` - результаты поиска по сетке

## Результаты

### SVM

Был выполнен поиск по сетке с перебором параметров предобработки данных и параметров моделей
Параметры сетки можно посмотреть в фаиле `config.py`

По параметрам предобработки был произведен полный перебор, по параметрам моделей - 
рандомизированный поиск с проверкой 500 наборов параметров

Смысл параметров предобработки можно посмотреть в `dataset.py`.

В результате поиска некоторые параметры оказались оптимальными для каждой из пяти целевых
переменных, чтобы не перегружать основную таблицу, перечислим их отдельно

<table>
    <caption>Общие значения параметров предобработки</caption>
    <thead>
        <td>Параметр</td><td>Значение</td>
    </thead>
    <tr><td>X1_num_log</td><td>True</td></tr>
    <tr><td>n_folds</td><td>5</td></tr>
    <tr><td>rm_19</td><td>True</td></tr>
    <tr><td>seed</td><td>42</td></tr>
</table>

<table>
    <caption>Общие значения параметров моделей</caption>
    <thead>
        <td>Параметр</td><td>Значение</td>
    </thead>
    <tr><td>class_weight</td><td>balanced</td></tr>
</table>

<table>
<caption>Параметры предобработки данных</caption>
<thead><td>Parameter</td><td>Target 1</td><td>Target 2</td><td>Target 3</td><td>Target 4</td><td>Target 5</td></thead>
<tr><td>X1_cat_to_bin</td><td>False</td><td>False</td><td>False</td><td>True</td><td>False</td></tr>
<tr><td>X1_num_std</td><td>False</td><td>True</td><td>False</td><td>False</td><td>False</td></tr>
<tr><td>X1_num_zero_one</td><td>False</td><td>False</td><td>False</td><td>True</td><td>False</td></tr>
<tr><td>compress_binary</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td></tr>
<tr><td>outliers_X1_num</td><td>False</td><td>True</td><td>True</td><td>True</td><td>False</td></tr>
<tr><td>rm_X1_cat_rare</td><td>True</td><td>True</td><td>False</td><td>False</td><td>False</td></tr>
</table>

<table>
<caption>Параметры моделей</caption>
<thead><td>Parameter</td><td>Target 1</td><td>Target 2</td><td>Target 3</td><td>Target 4</td><td>Target 5</td></thead>
<tr><td>C</td><td>10</td><td>0.1</td><td>0.1</td><td>1</td><td>0.1</td></tr>
<tr><td>coef0</td><td>0</td><td>0</td><td>0.01</td><td>0.01</td><td>0</td></tr>
<tr><td>degree</td><td>-</td><td>-</td><td>3</td><td>-</td><td>2</td></tr>
<tr><td>gamma</td><td>-</td><td>-</td><td>auto</td><td>auto</td><td>auto</td></tr>
<tr><td>kernel</td><td>linear</td><td>linear</td><td>poly</td><td>sigmoid</td><td>poly</td></tr>
<tr><td>shrinking</td><td>True</td><td>False</td><td>False</td><td>True</td><td>True</td></tr>
</table>

<table>
<caption>ROC AUC на локальной валидации</caption>
<thead><td>Parameter</td><td>Target 1</td><td>Target 2</td><td>Target 3</td><td>Target 4</td><td>Target 5</td></thead>
<tr><td>train mean</td><td>0.6162</td><td>0.6149</td><td>0.6825</td><td>0.5982</td><td>0.5895</td></tr>
<tr><td>train std</td><td>0.0013</td><td>0.0032</td><td>0.0031</td><td>0.0050</td><td>0.0064</td></tr>
<tr><td>valid mean</td><td>0.6011</td><td>0.6021</td><td>0.6289</td><td>0.6076</td><td>0.5573</td></tr>
<tr><td>valid std</td><td>0.0053</td><td>0.0132</td><td>0.0156</td><td>0.0186</td><td>0.0235</td></tr>
</table>

Ответ для одной целевой переменно получен в результате усреднения предсказаний соответствующей модели по фолдам.
