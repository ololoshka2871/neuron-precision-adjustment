# Точная настройка кварцевых резонаторов лазером 

## Описание
Используется эффект того, что частота резонанса резонатора зависит от массы двизущихся частей.
При производстве резонатора выделяются специальные области, покрытые серебром и не являющиеся частью электродной системы резонатора.
Если частично испарить серебро в данных областях, то частота резонанса резонатора увеличится.

## Некоторые начальные данные
* Чатота резонатора изменяется примерно на 1 Гц при изменении массы серебра на 3 нанограмма.
* Толщина слоя серебра ~0.5 мкм
* Мощность лазера 30 Вт
* Ограничение неточности установки резонатора в пределах рабчей области:
    - Смещние (X, Y): (+/- 0.3 мм, <= 0.5 мм)
    - Угол не более 10 градусов

## Эмпирические данные
* Коэфициент поглащения теплового излучения должет быть менее 0.05, иначе тепловая модель показывает очень большие температуры резонатора.

## Алгоритм работы

### Версия 1
Нейронная сеть должна заменять оператора лазерной обработки резонаторов. В качестве входных данных используются данные о текущей частоте резонатора измеренный с периодом ~0,4 с. Сразу оговорено, что нельзя использовать при обучении внутринние данные модели, поскольку когда сеть будет управлять реальным оборудованием такие данные получить будет невозможно.
Итого на вход нейронной сети передаются данные о текущем положении лазера, и иостория измения частот некоторй длины. Выходом являлись нейроны, данные с которых использовались для следующего движения лазера. Новая позиция, скорость и мощность лазера (0..1). Также для симуляции назначается глобальный таймаут чтобы выйти из цикла обучения.
Оценка приспособленности конкретного экземпляра нейронной сети (индивидуума) проводилась по следующим категориям:
- Относительная дистанция до целевой частоты - меньше - лучше
- Относителдьный диссбаланс - меньше (по модулю) - лучше
- Относительный штраф за попадание куда не надо - меньше - лучше
- Относительное время симуляции - меньше - лучше
- Точность самоошенки - больше - лучше
- Максимальная достигнутая температура - меньше - лучше
- Средняя скорость движения - больше - лучше
- Оценка за причину остановки - больше - лучше
Данная система быстро выявила неспособность адекватно управлять процессом и была заменена на вторую версию.

### Версия 2
Все то же самое что и перая версия, но на вход нейронная сеть получает не только историю измерения частоты, но и историю перемещений станка,
Кроме этого время симуляции (0..1) и текущий уровень "энергии". Энергия была придумана для того чтобы сподвигнуть нейронную сеть на движение в сторону цели. Если луч попадает в мишень, что начисляется бонус энергии, а за перемещения энергия отнимается. Таким образом нейронная сеть будет стараться двигаться в сторону цели, но не будет двигаться в случайном направлении (в теории).
Также были добавлены критерии оценки, но их влияние мало.
Вышеописаная система работает лучше, но как бы я не менял условия симуляции, всегда рано или поздно выбраный сетью алгоритм сводится к следующему:
Сделать 1 или несколько диоганальных проходов лазером (угол зависит от условий), а потом встать и завершить симуляцию. видимой реакции на историю измерений не выявлено. 
Хотя на самом деле нет! Последняя итерация версии 2 показывает, что сеть таки научилась с высокой долей достоверности угадывать когда следует "остановиться", поскольку конечное отклонеие частоты от целевой частоты стало очень малым.
Вцелом результат пока неудовлетворительный.

### Версия 3
Задача: Поскольку не удалось научить сеть управлять процессом напрямую, попробуем при помощи сети направлять вучной алгоритм и сделать предсказание момента остановки.
Вводные данные:
- Теперь вместо того чтобы получать от сети точку назначения, быдем алгоритмчески перемещать лазер зигзагом начиная с верзнего края рабочей области. 
- От сети будем получать параметрами такого движения: шаг по вертикали следующего прохода, скорость движения и мощность лазера на этом проходе.
- Как и в прошлых версиях выделим специальный нейрон самооценки, значение которого будет интерпритировать как блидость частоты к целевой частоте. если значение его станет меньше порога, резонатор считается настроеным.