import random
import sys
from os import path
from pathlib import Path
from typing import List

from sc2 import maps
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

sys.path.append("ares-sc2/src/ares")
sys.path.append("ares-sc2/src")
sys.path.append("ares-sc2")
sys.path.append("queens-sc2")



import yaml

from bot.main import MyBot
from ladder import run_ladder_game

# change if non default setup / linux
# if having issues with this, modify `map_list` below manually
MAPS_PATH: str = "C:\\Program Files (x86)\\StarCraft II\\Maps"
CONFIG_FILE: str = "config.yml"
MAP_FILE_EXT: str = "SC2Map"
MY_BOT_NAME: str = "MyBotName"
MY_BOT_RACE: str = "MyBotRace"


def main():
    bot_name: str = "MyBot"
    race: Race = Race.Zerg

    __user_config_location__: str = path.abspath(".")
    user_config_path: str = path.join(__user_config_location__, CONFIG_FILE)
    # attempt to get race and bot name from config file if they exist
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file)
            if MY_BOT_NAME in config:
                bot_name = config[MY_BOT_NAME]
            if MY_BOT_RACE in config:
                race = Race[config[MY_BOT_RACE].title()]

    bot1 = Bot(race, MyBot(), bot_name)

    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        # Descobre mapas instalados e restringe aos de interesse (patch 5.0.14)
        installed_maps: List[str] = [
            p.name.replace(f".{MAP_FILE_EXT}", "")
            for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
            if p.is_file()
        ]
        #allowed_5014 = {"IncorporealAIE_v4", "PersephoneAIE_v4", "PylonAIE_v4", "TorchesAIE_v4", "LeyLinesAIE_v3", "MagannathaAIE_v2", "UltraloveAIE_v2"}
        allowed_5014 = {"IncorporealAIE_v4"}
        map_list: List[str] = [m for m in installed_maps if m in allowed_5014]

        if not map_list:
            # fallback: tenta carregar explicitamente pelos nomes, caso não estejam listados no diretório padrão
            map_list = list(allowed_5014)
        # alternative example code if finding the map path is problematic
        # map_list: List[str] = [
        #     "BerlingradAIE",
        #     "InsideAndOutAIE",
        #     "MoondanceAIE",
        #     "StargazersAIE",
        #     "WaterfallAIE",
        #     "HardwireAIE",
        # ]

        random_race = random.choice([Race.Zerg])
        print("Starting local game...")
        chosen_map = random.choice(map_list)
        print(f"Selected map: {chosen_map}")


        run_game(
            maps.get(chosen_map),
            [
                bot1,
                Computer(random_race, Difficulty.CheatVision, ai_build=AIBuild.Rush),
                #Builds: RandomBuild, Rush, Timing, Power, Macro, Air 
                #Computer(random_race, Difficulty.CheatVision, ai_build=AIBuild.RandomBuild),
            ],
            realtime=False,

        )


# Start game
if __name__ == "__main__":
    main()
