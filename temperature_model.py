#!/usr/bin/env python


class TemperatureModel:
    """
    Простая модель температуры объекта.
    Моделирует нагрев и охлаждение объекта во времени зная теплоёмкость и температуру окружающкй среды.
    Объкут поглащает энергию и отдает её окружающей среде, скорость зависит от разнцы температур.
    """

    def __init__(self, heat_capacity: float, heat_dissipation_rate: float,
                 ambient_temperature=273.0, temperature: float | None = None,
                 strike_dissipation: float = 0.95):
        """
        :param heat_capacity: Теплоёмкость объекта [J / K]
        :param heat_dissipation_rate: Скорость рассеяние энергии в окружающую стреду [0..1]
        :param ambient_temperature: Температура окружающей среды [K]
        :param temperature: Текущая температура объекта, если не указана, то равна температуре окружающей среды [K]
        :param strike_dissipation: Доля энергии рассеиваемая непосредственно при ударе лазера [0..1]
        """
        assert (heat_dissipation_rate <=
                1.0 and heat_dissipation_rate >= 0.0)
        assert (strike_dissipation <= 1.0 and strike_dissipation >= 0.0)
        assert (ambient_temperature >= 0.0)

        self._heat_dissipation_rate = heat_dissipation_rate
        self._heat_capacity = heat_capacity  # [K]
        current_temperature = temperature if temperature is not None else ambient_temperature
        self._current_energy = current_temperature * heat_capacity  # [xJ]
        self._consume_rate = 1.0 - strike_dissipation
        self._ambient_temperature = ambient_temperature

        self._minimal_energy = self._ambient_temperature * self._heat_capacity

    def current_temperature(self) -> float:
        """
        Текущая температура объекта [K]
        """
        return self._current_energy / self._heat_capacity

    def tick(self, power: float = 0.0, dt: float = 1.0):
        """
        Производит шаг моделирования во времени.
        :param power: Мощность, которую объект получил за шаг [W]
        :param dt: Время шага [s]
        """

        consumed_energy = power * self._consume_rate * dt # [J]
        self._current_energy += consumed_energy
        energy_dispersed = (self._current_energy - self._minimal_energy) * self._heat_dissipation_rate * dt
        self._current_energy -= energy_dispersed

    @property
    def ambient_temperature(self) -> float:
        return self._ambient_temperature


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    QUARTZ_HEAT_CAPACITY = 750.0  # [J / (kg * K)]
    QUARTZ_DESTENSITY = 2650.0  # [kg / m^3]
    REZONATOR_VOLUME = 0.9 * pow(1e-3, 3)  # [mm^3] -> [m^3]
    MODEL_TICK_TIME = 0.01  # [s]

    HEAT_DISSIPATION_SPEED = 0.9  # [J / (K * s)]

    LASER_POWER = 30.0  # [W]

    f, ax = plt.subplots(1, 1)

    rezonator_mass = QUARTZ_DESTENSITY * REZONATOR_VOLUME
    m = TemperatureModel(heat_capacity=rezonator_mass * QUARTZ_HEAT_CAPACITY,
                         heat_dissipation_rate=HEAT_DISSIPATION_SPEED)

    data_x = [0.0]
    data_y = [m.current_temperature()]

    curve, = ax.plot(data_x, 'ro-')

    plt.show(block=False)

    while True:
        click = f.ginput(timeout=MODEL_TICK_TIME)
        if len(click) == 0:
            energy = 0.0
        else:
            energy = LASER_POWER

        m.tick(power=energy, dt=MODEL_TICK_TIME)

        data_x.append(data_x[-1] + MODEL_TICK_TIME)
        t = m.current_temperature()
        print(f"T = {t:.10f}")
        data_y.append(t)

        curve.set_data(data_x, data_y)
        ax.relim()
        ax.autoscale_view()
        f.canvas.draw_idle()
