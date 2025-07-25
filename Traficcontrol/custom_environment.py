
from sumo_rl import SumoEnvironment


class CustomEnvironment:

    def __init__(self,
                 route_file: str,
                 gui: bool,
                 num_seconds: int,
                 min_green: int,
                 max_green: int,
                 yellow_time: int,
                 delta_time: int) -> None:
        
        self.route_file = route_file
        self.gui = gui
        self.num_seconds = num_seconds
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.delta_time = delta_time

    def get_sumo_env(self, fixed: bool) -> SumoEnvironment:
       
        return SumoEnvironment(
            net_file="interseccion/k23c16.net.xml",
            route_file=self.route_file,
            use_gui=self.gui,
            num_seconds=self.num_seconds,
            delta_time=self.delta_time,
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            max_green=self.max_green,
            fixed_ts=fixed,
            add_per_agent_info=False,
            sumo_warnings=False,
            single_agent=True,
        )
