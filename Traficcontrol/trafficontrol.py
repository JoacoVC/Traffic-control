import yaml
from runner import Runner
from plotter import Plotter

config_file = 'configs/test.yaml'
with open(config_file) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Validación sencilla (opcional)
assert 'Plotter_settings' in config
assert 'Agent_settings' in config

p = Plotter()
p.set_configs(config['Plotter_settings'])


r = Runner(config['Agent_settings'], p)
r.run()

