## Autores
Leober Arturo Duran Raudales - lduranr@unal.edu.co

José Joaquín Vergara Cartagena - jvergarac@unal.edu.co

Universidad Nacional de Colombia, Sede La Paz

# Q-Learning y SARSA para la Gestión Inteligente del Tráfico Semafórico

Este proyecto implementa y compara algoritmos de **Aprendizaje por Refuerzo** (Q-Learning y SARSA) como alternativa a los semáforos de ciclos fijos, con el objetivo de **mejorar el flujo vehicular urbano** mediante una gestión semafórica adaptativa. Se utilizan entornos simulados en **SUMO (Simulation of Urban Mobility)** y la librería **sumo-rl** para crear, entrenar y evaluar los modelos.

## Tecnologías y Herramientas

- Python
- [SUMO](https://www.eclipse.org/sumo/) - Simulador de tráfico urbano
- [sumo-rl](https://github.com/LucasAlegre/sumo-rl) - Librería de entrenamiento para agentes RL en SUMO
- Q-Learning (off-policy)
- True Online SARSA(λ) (on-policy con aproximación de Fourier)

## Objetivos del Proyecto

- Evaluar si los algoritmos de RL pueden adaptarse mejor que los semáforos tradicionales.
- Medir el desempeño usando métricas como **tiempo de espera** y **vehículos detenidos**.
- Probar el rendimiento en diferentes escenarios urbanos simulados.



## Modelado de la Intersección

- Cruce simulado: **Carrera 23 con Calle 16, Valledupar (Colombia)**.
- Extraído desde OpenStreetMap y editado con NetEdit (SUMO).
- Intersección con 8 vías de distinta capacidad y flujo.

## Configuración del Entorno

```python
SumoEnvironment(
    net_file='k23c16.net.xml',
    route_file='trafic.rou.xml',
    use_gui=False,
    num_seconds=100000,
    delta_time=5,
    yellow_time=2,
    min_green=5,
    max_green=60,
    single_agent=True
)


## Algoritmos Implementados
Q-Learning
Off-policy

Actualiza Q(s, a) con la acción óptima futura

Mejor rendimiento en tráfico bajo

SARSA (True Online SARSA(λ))
On-policy

Aproximación de funciones con Fourier

Mejor adaptación en tráfico complejo

## Representación del Estado
Fase activa (one-hot)

Tiempo mínimo en verde

Densidad y longitud de colas por carril

Ejemplo: [0, 1, 0, ..., 0.07, 0.06, 0.08]

## Función de Recompensa
Se usó diff-waiting-time:

python
Copiar
Editar
reward = W(t-1) - W(t)
Minimiza el tiempo de espera acumulado por los vehículos.

## Experimentos
Se compararon cinco agentes:

Q-Learning Run 1 y Run 2

SARSA Run 1 y Run 2

Semáforo tradicional de ciclos fijos






