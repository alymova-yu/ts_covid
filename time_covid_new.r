# Предсказание заболеваемости новой короновирусной инфекцией
# в Москве

# Подключение библиотек
library(rio)
library(tidyverse)
library(fpp3)
library(caret)
library(ranger)
library(forecast)
library(e1071)
library(hydroGOF)


# Импорт данных
ds=import('табл1.xlsx')
head(ds)

# Выбор колонок с датами, регионом и ежедневным числом новых заражений
ds=ds[c(1:2, 7)]
colnames(ds)[1:3] = c('date', 'region', 'infections')
glimpse(ds)
sapply(ds, function (x) sum(is.na (x)))

# Преобразование во временной ряд, фильтрация по региону
ts = as_tsibble(ds, index = date, key=region, regular = FALSE)
ts_m = filter(ts, region == "Москва")
ts_mm = mutate(ts_m, date = as.Date(date))
ts_mm = ts_mm[,c(1,3)]
head(ts_mm)

# Отображение ряда, ACF, нарезки по годам
gg_tsdisplay(ts_mm)

# Разложение на тренд, сезонность и остаток (STL)
stl_model = model(ts_mm, stl = STL(infections ~ trend(window=14) +
                                     season(window=180)))
components(stl_model) %>% autoplot()

# Разделение данных на тренировочную и тестовую выборки
train = filter(ts_mm, date < ymd('2022-06-01'))
test = filter(ts_mm, date >= ymd('2022-06-01'))

# Обучение наивной модели, ETS, тета-модели и ARIMA
models = model(train, naive = NAIVE(infections),
               ets = ETS(infections),
               theta = THETA(infections),
               arima = ARIMA(infections),
               arima_f = ARIMA(infections ~ fourier(K=3) + PDQ(0,0,0)))

# Предсказание числа новых заболеваний на 6 месяев, сравнение с тестовой выборкой
fcst = forecast(models, h='6 months')

# Сравнение прогнозов
fabletools::accuracy(fcst, test)

# Визуализация прогнозов моделей с наилучшим RMSE
fcst_sub = filter(fcst, .model %in% c('ets', 'arima_f'))
autoplot(fcst_sub, test)

# Характеристики ETS-модели, подобранные автоматически
models$ets

# Проверка тех же моделей кросс-валидацией скользящим окном
ts_slide = slide_tsibble(ts_mm, .size = 180, .step = 30)
unique(ts_slide$.id)

models_slide = model(ts_slide, naive = NAIVE(infections),
               ets = ETS(infections),
               theta = THETA(infections),
               arima = ARIMA(infections),
               arima_f = ARIMA(infections ~ fourier(K=3) + PDQ(0,0,0)))

fcst_slide = forecast(models_slide, h='1 months')

fabletools::accuracy(fcst_slide, ts_mm)


### Использование для здачи предсказания моделей машинного обучения

## Подготовка предикторов 

# Создание тригонометрических сезонных предикторов
fourier_x = fourier(ts_mm, K=3)
colnames(fourier_x) = c('sin1', 'cos1', 'sin2', 'cos2',
                        'sin3', 'cos3')

# Объединение набора даных, добавление предиктора времени
ts_m2 = bind_cols(mutate(ts_mm, t = 1:nrow(ts_mm)), fourier_x)

# Добавление предикторов, несущих информацию о числе заражений 6-7 месяцев назад
hal = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  hal = c(hal, (ts_m2$infections[x-182]))}

hal_h = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  hal_h = c(hal_h, (ts_m2$infections[x-196]))}

hal_m = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  hal_m = c(hal_m, (ts_m2$infections[x-212]))}

lag1 = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  lag1 = c(lag1, (ts_m2$infections[x-182])-(ts_m2$infections[x-191]))}

lag2 = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  lag2 = c(lag2, (ts_m2$infections[x-191])-(ts_m2$infections[x-198]))}

lag3 = rep(NA, times=212)
for(x in 213:nrow(ts_m2)) {
  lag3 = c(lag3, (ts_m2$infections[x-198])-(ts_m2$infections[x-212]))}

# Подготовка наборов данных
# С одним типом предикторов

ts_hal = bind_cols(ts_m2, hal = hal,
                   hal_h = hal_h,
                   hal_m = hal_m)

ts_hal = ts_hal[213:nrow(ts_hal),]
head(ts_hal)

# С другим типом предикторов
ts_lag = bind_cols(ts_m2, 
                   lag1 = lag1,
                   lag2 = lag2,
                   lag3=lag3)

ts_lag = ts_lag[213:nrow(ts_lag),]
head(ts_lag)

# Со всеми предикторами

ts_hl = bind_cols(ts_m2, hal = hal,
                   hal_h = hal_h,
                   hal_m = hal_m,
                   lag1 = lag1,
                   lag2 = lag2,
                   lag3=lag3)

ts_hl = ts_hl[213:nrow(ts_hl),]
head(ts_hl)

# Разделение данных на тренировочные и тестовые выборки 
# для тестовых выборок отсекаются последние 6 месяцев наблюдений
train_hal = filter(ts_hal, date < ymd('2022-06-01'))
test_hal = filter(ts_hal, date >= ymd('2022-06-01'))

train_lag = filter(ts_lag, date < ymd('2022-06-01'))
test_lag = filter(ts_lag, date >= ymd('2022-06-01'))

train_hl = filter(ts_hl, date < ymd('2022-06-01'))
test_hl = filter(ts_hl, date >= ymd('2022-06-01'))
# 
cv_params = trainControl(method = 'cv', number = 5)

# Обучение моделей
ls_hal = train(infections ~ .,
               data = train_hal, 
               trControl = cv_params,
               method = 'lm')
ls_lag = train(infections ~ .,
               data = train_lag, 
               trControl = cv_params,
               method = 'lm')
ls_hl = train(infections ~ .,
               data = train_hl, 
               trControl = cv_params,
               method = 'lm')

rf_hal = train(infections ~.,
           data = train_hal,  
            trControl = cv_params, 
            method = 'ranger',
            num.trees = 1000)
rf_lag = train(infections ~.,
           data = train_lag,  
           trControl = cv_params, 
           method = 'ranger',
           num.trees = 1000)
rf_hl = train(infections ~.,
           data = train_hl,  
           trControl = cv_params, 
           method = 'ranger',
           num.trees = 1000)

gb_hal = train(infections ~.,
           data = train_hal, 
           trControl = cv_params,
           method = 'xgbTree')
gb_lag = train(infections ~.,
           data = train_lag, 
           trControl = cv_params,
           method = 'xgbTree')
gb_hl = train(infections ~.,
           data = train_hl, 
           trControl = cv_params,
           method = 'xgbTree')

svm_hal = svm(infections ~.,
          data = train_hal)
svm_lag = svm(infections ~.,
          data = train_lag)
svm_hl = svm(infections ~.,
          data = train_hl)

# предсказания моделей
ls_fcst_hal = predict(ls_hal, test_hal)
ls_fcst_lag = predict(ls_lag, test_lag)
ls_fcst_hl = predict(ls_hl, test_hl)

rf_fcst_hal = predict(rf_hal, test_hal)
rf_fcst_lag = predict(rf_lag, test_lag)
rf_fcst_hl = predict(rf_hl, test_hl)

gb_fcst_hal = predict(gb_hal, test_hal)
gb_fcst_lag = predict(gb_lag, test_lag)
gb_fcst_hl = predict(gb_hl, test_hl)

svm_fcst_hal = predict(svm_hal, test_hal)
svm_fcst_lag = predict(svm_lag, test_lag)
svm_fcst_hl = predict(svm_hl, test_hl)

# Оценка качества предсказаний
RMSE_ls_hal=rmse(ls_fcst_hal, test_lag$infections)
RMSE_ls_lag=rmse(ls_fcst_lag, test_lag$infections)
RMSE_ls_hl=rmse(ls_fcst_hl, test_lag$infections)

RMSE_rf_hal=rmse(rf_fcst_hal, test_lag$infections)
RMSE_rf_lag=rmse(rf_fcst_lag, test_lag$infections)
RMSE_rf_hl=rmse(rf_fcst_hl, test_lag$infections)

RMSE_gb_hal=rmse(gb_fcst_hal, test_lag$infections)
RMSE_gb_lag=rmse(gb_fcst_lag, test_lag$infections)
RMSE_gb_hl=rmse(gb_fcst_hl, test_lag$infections)

RMSE_svm_hal=rmse(svm_fcst_hal, test_lag$infections)
RMSE_svm_lag=rmse(svm_fcst_lag, test_lag$infections)
RMSE_svm_hl=rmse(svm_fcst_hl, test_lag$infections)

metrics_lag = data.frame(svm = c(RMSE_svm_hal, RMSE_svm_lag, RMSE_svm_hl),
                         ls = c(RMSE_ls_hal, RMSE_ls_lag, RMSE_ls_hl),
                         rf = c(RMSE_rf_hal, RMSE_rf_lag, RMSE_rf_hl),
                         gb = c(RMSE_gb_hal, RMSE_gb_lag, RMSE_gb_hl))
row.names(metrics_lag) = c('hal', 'lag', 'hl')
# Таблица полученных оценок
metrics_lag
# Минимальное значение RMSE
min(metrics_lag)

# Изображение реальных данных и предсказаний моделей с лучшими оценками
plot(test_lag$date, test_lag$infections, type = 'l', 
     xlab = 'Date', ylab = 'Infections', 
     main = "Predictions based on infection numbers")
lines(test_lag$date, rf_fcst_hl,  col='red')
lines(test_lag$date, svm_fcst_hl,  col='blue')



## Добавление новых предикторов

# Загрузка предикторов из второго набора данных
pred = import('табл2.xlsx')
pred = pred[2:4]
colnames(pred) = c('date', 'marker', 'num')

# Предобработка данных
ts_pred = as_tsibble(pred, index = date, key=marker, regular = FALSE)
ts_pred = mutate(ts_pred, date = ymd(as.Date(date)))

# Проверка совпадения дат в наборах данных
head(ts_mm$date, n=1) - head(ts_pred$date, n=1)
tail(ts_mm$date, n=1) - tail(ts_pred$date, n=1)

# Разделение набора данных на основе маркера
new = split(ts_pred,ts_pred$marker)
names = c('date', unique(ts_pred$marker))
pred_new = bind_cols(new[[1]]$date, 
                     new[[1]]$num) 
for(i in 2:length(new)){
  pred_new = bind_cols(pred_new,
                       new[[i]]$num)
}

# Отсечение и вменение данных для выравнивания с первым набором данных
colnames(pred_new) = names
pred_new = filter(pred_new, date >= ymd(head(ts_mm$date, n=1)))
ts_ps = c(pred_new[[1]], rep(0, times=65))
for (x in 2:length(pred_new)){
  a = rep(0, times=65)
  ts_ps = bind_cols(ts_ps, c(pred_new[[x]], a))
}
colnames(ts_ps) = names

# Использование данных для конструирования предиктора
tss = ts_ps$date
for (i in 2:length(pred_new)){
  n = rep(NA, times=189)
  for(x in 190:nrow(ts_ps)) {n = c(n, (ts_ps[[i]][x-182])-(ts_ps[[i]][x-189]))
  
  }
  tss = bind_cols(tss, n)
}
colnames(tss) = names

# Подготовка датасетов для обучения 
# только с новыми предикторами
ts_n = bind_cols(ts_m2, 
                tss[2:16])
ts_n = ts_n[213:nrow(ts_n),]
# со всеми предикторами
ts_hln = bind_cols(ts_m2, hal = hal,
                hal_h = hal_h,
                hal_m = hal_m,
                lag1 = lag1,
                lag2 = lag2,
                lag3=lag3,
                tss[2:16])
ts_hln = ts_hln[213:nrow(ts_hln),]

# Разделение данных на тренировочные и тестовые выборки
train_n = filter(ts_n, date < ymd('2022-06-01'))
test_n = filter(ts_n, date >= ymd('2022-06-01'))

train_hln = filter(ts_hln, date < ymd('2022-06-01'))
test_hln = filter(ts_hln, date >= ymd('2022-06-01'))


# Обучение моделей
ls_n = train(infections ~ .,
               data = train_n, 
               trControl = cv_params,
               method = 'lm')
ls_hln = train(infections ~ .,
              data = train_hln, 
              trControl = cv_params,
              method = 'lm')

rf_n = train(infections ~.,
               data = train_n,  
               trControl = cv_params, 
               method = 'ranger',
               num.trees = 1000)
rf_hln = train(infections ~.,
              data = train_hln,  
              trControl = cv_params, 
              method = 'ranger',
              num.trees = 1000)

gb_n = train(infections ~.,
               data = train_n, 
               trControl = cv_params,
               method = 'xgbTree')
gb_hln = train(infections ~.,
              data = train_hln, 
              trControl = cv_params,
              method = 'xgbTree')

svm_n = svm(infections ~.,
              data = train_n)
svm_hln = svm(infections ~.,
             data = train_hln)

# предсказания моделей
ls_fcst_n = predict(ls_n, test_n)
ls_fcst_hln = predict(ls_hln, test_hln)

rf_fcst_n = predict(rf_n, test_n)
rf_fcst_hln = predict(rf_hln, test_hln)

gb_fcst_n = predict(gb_n, test_n)
gb_fcst_hln = predict(gb_hln, test_hln)

svm_fcst_n = predict(svm_n, test_n)
svm_fcst_hln = predict(svm_hln, test_hln)

# Оценка качества предсказаний
RMSE_ls_n=rmse(ls_fcst_n, test_lag$infections)
RMSE_ls_hln=rmse(ls_fcst_hln, test_lag$infections)

RMSE_rf_n=rmse(rf_fcst_n, test_lag$infections)
RMSE_rf_hln=rmse(rf_fcst_hln, test_lag$infections)

RMSE_gb_n=rmse(gb_fcst_n, test_lag$infections)
RMSE_gb_hln=rmse(gb_fcst_hln, test_lag$infections)

RMSE_svm_n=rmse(svm_fcst_n, test_lag$infections)
RMSE_svm_hln=rmse(svm_fcst_hln, test_lag$infections)

metrics_lag[nrow(metrics_lag) + 1, ] = c(RMSE_svm_n, RMSE_ls_n, RMSE_rf_n, RMSE_gb_n)
metrics_lag[nrow(metrics_lag) + 1, ] = c(RMSE_svm_hln, RMSE_ls_hln, RMSE_rf_hln, RMSE_gb_hln)

row.names(metrics_lag) = c('hal', 'lag', 'hl', 'n', 'hln')
# Таблица полученных оценок
print(metrics_lag)
# Минимальное значение RMSE
min(metrics_lag)

# Изображение реальных данных и предсказаний моделей с лучшими оценками
plot(test_lag$date, test_lag$infections, type = 'l', 
     xlab = 'Date', ylab = 'Infections', 
     main = "Predictions based on infection numbers and markers")
lines(test_lag$date, rf_fcst_hln,  col='red')
lines(test_lag$date, svm_fcst_hln,  col='blue')


