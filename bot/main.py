"""
---------------------------
BOT CLICADINHA
---------------------------

Made of with Ares Random Example Bot

https://github.com/AresSC2/ares-random-example


Using the Queens framework

https://github.com/raspersc2/queens-sc2

"""


import math
from itertools import cycle
from typing import Dict, Optional, Set

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    UseAbility,
    UseAOEAbility,
    AttackTarget,
)
from ares.behaviors.macro import AutoSupply, Mining, SpawnController, GasBuildingController, BuildWorkers, ExpansionController, MacroPlan, BuildStructure, TechUp
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole, UnitTreeQueryType, BuildingPurpose, BuildingSize
from cython_extensions import cy_closest_to, cy_in_attack_range, cy_pick_enemy_target, cy_distance_to
from sc2 import unit
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
import time

from queens_sc2.queens import Queens



#_______________________________________________________________________________________________________________________
#          ARMY COMPOSITION
#_______________________________________________________________________________________________________________________


# Wrapper to prevent training workers while any spawn inhibitor is active
# Keeps logic local to this bot so we don't modify ares internals.
class BuildWorkersNoExpand(BuildWorkers):
    """Variant of BuildWorkers that doesn't execute when any spawn inhibitor
    is present.

    This ensures drone production is paused whenever the bot is saving
    resources for tech, upgrades, or any other inhibited macro state.
    """

    def execute(self, ai: "AresBot", config: dict, mediator) -> bool:
        inhibitors = getattr(ai, "spawn_inhibitors", set())
        if inhibitors:
            return False
        return super().execute(ai, config, mediator)

# this will be used for ares SpawnController behavior

# against Terran
ARMY_COMP_HYDRALING: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ZERGLING: {"proportion": 0.8, "priority": 0},
        UnitID.HYDRALISK: {"proportion": 0.2, "priority": 1},
    }
}

# against Protoss
ARMY_COMP_LING: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ZERGLING: {"proportion": 1.0, "priority": 0},
    }
}

# against other races
ARMY_COMP_ROACH: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ROACH: {"proportion": 1.0, "priority": 0},
    }
}

# against other races
ARMY_COMP_ROACHINFESTOR: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ROACH: {"proportion": 0.91, "priority": 1},
        UnitID.INFESTOR: {"proportion": 0.09, "priority": 0},
    }
}

# against other races
ARMY_COMP_ROACHCORRUPTOR: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ROACH: {"proportion": 0.61, "priority": 1},
        UnitID.CORRUPTOR: {"proportion": 0.3, "priority": 0},
        UnitID.INFESTOR: {"proportion": 0.09, "priority": 0},
    }
}



# against other races
ARMY_COMP_LINGROACH: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ZERGLING: {"proportion": 0.6, "priority": 1},
        UnitID.ROACH: {"proportion": 0.4, "priority": 0},

    }
}

# against terran
ARMY_COMP_MUTAROACH: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.MUTALISK: {"proportion": 0.9, "priority": 0},
        UnitID.ROACH: {"proportion": 0.1, "priority": 1 },

    }
}


# against flying units
ARMY_COMP_MUTAlLISK: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.MUTALISK: {"proportion": 1.0, "priority": 0},

    }
}







COMMON_UNIT_IGNORE_TYPES: set[UnitID] = {
    UnitID.EGG,
    UnitID.LARVA,
    UnitID.CREEPTUMORBURROWED,
    UnitID.CREEPTUMORQUEEN,
    UnitID.CREEPTUMOR,
}


class MyBot(AresBot):
    expansions_generator: cycle
    current_base_target: Point2
    _begin_attack_at_supply: float
    BURROW_AT_HEALTH_PERC: float = 0.35
    UNBURROW_AT_HEALTH_PERC: float = 0.9
    last_debug_time = 0
    
    # instance of the queens class
    queens: Queens

    def __init__(self, game_step_override: Optional[int] = None):
        

        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        if not getattr(Units, "_bool_patched", False):
            Units.__bool__ = lambda self: bool(self.amount)
            Units._bool_patched = True
        self.tag_worker_build_2nd_base = 0
        self.tag_worker_build_roach_warren = 0
        self.tag_worker_build_hydra_den = 0
        self.tag_worker_build_spine_crawler = 0
        self.tag_worker_build_2nd_spine_crawler = 0
        self.tag_worker_build_3rd_spine_crawler = 0
        self.tag_worker_second_gas = 0
        self.overlord_retreated = False
        self.spineCrawlerCheeseDetected = False
        self.reaperFound = False
        self.bansheeFound = False
        self.spore_workers: dict = {}  # base_tag -> worker_tag
        self._spore_extra_drones_done: set[int] = set()  # base tags where 2 extra drones were already trained
        self.random_race_discovered = False
        self.one_proxy_barracks_found = False
        self.two_proxy_barracks_found = False
        self.mutalisksFound = False
        self.proxy_pylon_found = False        
        self.one_proxy_gateWay_found = False
        self.two_proxy_gateWay_found = False
        self.photon_cannon_found = False
        self.terran_flying_structures = False
        self.tag_worker_build_spire = 0
        self._spore_bc_last_dispatch: dict[int, float] = {}  # base_tag -> game_time of last dispatch
        self._spore_bc_extra_drones_done: set[int] = set()  # base tags where 2 extra drones were already trained
        self.is_roach_attacking = False
        self.defending = False
        self.liberatorFound = False
        self.spawn_inhibitors: set[str] = set()
        self.speedMiningOn = True
        self.enemy_has_3_bases = False
        self.scout_targets = {}  # Dicionário para armazenar os alvos dos scouts
        self.mutalisk_targets = {}  # Dicionário para armazenar os alvos dos mutalisks
        self.enemies_on_creep = {}  # Dicionário para armazenar as unidades inimigas que estão no creep
        self.enemy_went_worker_rush = False
        self.enemy_went_ling_rush = False
        self.bo_changed = False
        self.my_overlords = {}
        self.stop_getting_gas = False
        self.workers_for_gas = 3
        self.tag_second_overlord = 0
        self.my_roaches = {}
        self.enemy_widow_mines = {}
        self.mid_game = False
        self.late_game = False
        self.mid_game_expansion_done = False
        self.tag_worker_infestation_pit = 0
        self.taf_worker_build_macro_hatch = 0
        self.macro_hatch_pos: Optional[Point2] = None
        self.second_base_canceled = False
        self.enemy_battlecruisers = {}
        self.enemy_banshees = {}
        self.last_known_banshee_positions = {}  # tag -> Point2, para seguir banshees invisíveis
        self.last_known_banshee_frames = {}     # tag -> game_loop frame quando banshee foi vista
        self.overseer_banshee_assignments = {}  # overseer_tag -> banshee_tag
        self.nydus_spot_set = False
        self.tag_third_overlord = 0
        self.tag_fourth_overlord = 0
        self.third_overlord_retreated = False
        self.fourth_overlord_retreated = False
        self.enemy_nat_cc_found = False
        self.scout_changeling_spawned = False
        self.infestation_pit_ordered = False
        self.evolution_chamber_ordered = False
        self.spire_ordered = False
        self.first_overlord = 0
        self._retreat_issued: bool = False
        self._retreating: bool = False
        self._evo_worker_tag: int = 0
        self.late_game_expansion_done = False
        self._proxy_stargate_queen_policy_applied: bool = False

        self._commenced_attack: bool = False


        self.creep_queen_tags: Set[int] = set()
        self.other_queen_tags: Set[int] = set()
        self.max_creep_queens: int = 4
        self._used_tumors: Set[int] = set()


        self.macro_plan = MacroPlan()
        self._queen_attack_target: Optional[int] = None
        self._gas_count_registered: int = 0

    def _apply_proxy_stargate_queen_policy(self) -> None:
        """Switch queens to a defensive policy against Proxy Stargate."""
        if self._proxy_stargate_queen_policy_applied:
            return

        anti_proxy_stargate_policy: Dict = {
            "creep_queens": {
                "active": False,
                "priority": 0,
                "max": 0,
                "defend_against_air": False,
                "defend_against_ground": False,
                "first_tumor_position": self.mediator.get_own_nat.towards(self.game_info.map_center, 9),
            },
            "inject_queens": {
                "active": True,
                "priority": 1,
            },
            "defence_queens": {
                "active": True,
            },
        }

        self.creep_queen_policy = anti_proxy_stargate_policy
        if hasattr(self, "queens") and self.queens is not None:
            self.queens.set_new_policy(self.creep_queen_policy, reset_roles=True)
            self._proxy_stargate_queen_policy_applied = True

    async def _hold_queens_near_hatcheries_vs_proxy_stargate(
        self, queens: Units, iteration: int
    ) -> None:
        """Keep queens close to own hatcheries while defending Proxy Stargate."""
        if "Proxy_Stargate" not in self.enemy_strategy or not queens:
            return

        # Throttle movement orders to avoid command spam.
        if iteration % 8 != 0:
            return

        ready_bases: Units = self.townhalls.ready
        for queen in queens:
            if ready_bases:
                anchor_base: Unit = ready_bases.closest_to(queen.position)
                anchor_pos: Point2 = anchor_base.position.towards(self.game_info.map_center, 2)
            elif self.first_base is not None:
                anchor_pos = self.first_base.position.towards(self.game_info.map_center, 2)
            else:
                anchor_pos = self.start_location

            if queen.distance_to(anchor_pos) > 8:
                self.do(queen.move(anchor_pos))


    @property
    def attack_target(self) -> Point2:
        ground_structures = self.enemy_structures.filter(lambda s: not s.is_flying)
        if ground_structures:
            # using a faster cython alternative here, see docs for all available functions
            # https://aressc2.github.io/ares-sc2/api_reference/cython_extensions/index.html
            return cy_closest_to(self.start_location, ground_structures).position
        # not seen anything in early game, just head to enemy spawn
        elif self.time < 240.0:
            return self.enemy_start_locations[0]
        # else search the map
        else:
            # cycle through expansion locations
            if self.is_visible(self.current_base_target):
                self.current_base_target = next(self.expansions_generator)

            return self.current_base_target

    def _prepare_step(self, state, proto_game_info) -> None:
        """Compatibility shim for python-sc2 versions that call _prepare_step synchronously."""
        self.state = state
        loop: int = state.game_loop
        if self.realtime and self.last_game_loop + 4 > loop and loop != 0:
            return

        self.last_game_loop = loop
        return super(AresBot, self)._prepare_step(state, proto_game_info)
        
#_______________________________________________________________________________________________________________________
#          ON START
#_______________________________________________________________________________________________________________________

    async def on_start(self) -> None:
        await super(MyBot, self).on_start()

        # Register macro behaviors now that behavior_executioner is available
        self.register_behavior(self.macro_plan)
    
        self.EnemyRace = self.enemy_race  
        self.rally_point_set = False
        self.first_base = self.townhalls.first if self.townhalls else None
        self.second_base = None
        overlords: Units = self.units(UnitID.OVERLORD)
        self.first_overlord = overlords.first if overlords else 0
        self.worker_scout_tag = 0
        self.enemy_strategy = []

    
        self.current_base_target = self.enemy_start_locations[0]
        self.expansions_generator = cycle(
            [pos for pos in self.expansion_locations_list]
        )



        self.creep_queen_policy: Dict = {
            "creep_queens": {
                "active": True,
                "priority": 0,
                "max": 20,
                "defend_against_air": True,
                "defend_against_ground": True,
                "first_tumor_position": self.mediator.get_own_nat.towards(self.game_info.map_center, 9),
            },
            "inject_queens": {
                "active": True,
                "priority": 1,
            },
            "defence_queens": {"active": False},
        }




        # Find the ID of the opponent    
        self.opponent = self.opponent_id
        if self.opponent_id is not None:
            await self.chat_send(self.opponent_id)
            print("The opponent ID is: ")
            print(self.opponent_id)
        else:
            print("Warning: opponent_id is None, cannot send chat message.")
    
        # BotKiller
        if self.opponent_id == "da0fe671-3f51-4c48-8ac2-252cb67ee545":
            self._begin_attack_at_supply = 1
    
    
        # LiShiMinV2
        elif self.opponent_id == "0d0d9c44-2520-457d-84ba-7f6ffe167a3e":
            self._begin_attack_at_supply = 1
    

        else:
            if self.EnemyRace == Race.Terran:
                self._begin_attack_at_supply = 40
    
            elif self.EnemyRace == Race.Protoss:
                self._begin_attack_at_supply = 10
    
            elif self.EnemyRace == Race.Zerg:
                self._begin_attack_at_supply = 30
    
            elif self.EnemyRace == Race.Random:
                self._begin_attack_at_supply = 30
    
        # Initialize the queens class
        self.queens = Queens(
            self, queen_policy=self.creep_queen_policy
        )

        # Instância única do Mining behavior, criada UMA VEZ aqui no on_start.
        # IMPORTANTE: NÃO criar uma nova instância a cada frame em _macro().
        # O Mining possui caches internos (locked_action_tags, safe_long_distance_mineral_fields)
        # que evitam cálculos pesados a cada step. Se a instância for recriada toda frame,
        # esses caches são zerados e o método _safe_long_distance_mineral_fields() (O(n) nos
        # minerais do mapa) é reexecutado para CADA worker sem recurso — custo que cresce
        # significativamente após ~15 minutos, quando patches começam a esgotar.
        self.mining_behavior: Mining = Mining(workers_per_gas=self.workers_for_gas)
    
        # Send Overlord to scout on the second base
        await self.send_overlord_to_scout()

#_______________________________________________________________________________________________________________________
#          SEND OVERLORD TO SCOUT
#_______________________________________________________________________________________________________________________


    async def send_overlord_to_scout(self):
        # Select the first Overlord
        overlords: Units = self.units(UnitID.OVERLORD)
        if not overlords:
            return

        overlord = overlords.first
        self.first_overlord = overlord
    
        enemy_natural_location = self.mediator.get_enemy_nat
        target = enemy_natural_location.position.towards(self.game_info.map_center, 12)

        if self.EnemyRace == Race.Terran:
            # vs Terran: apenas vai para o target, sem ir para hg_spot
            self.do(overlord.move(target))
        else:
            # vs outras raças: vai para o target e depois para hg_spot
            self.do(overlord.move(target))
            hg_spot = self.mediator.get_closest_overlord_spot(
                from_pos=enemy_natural_location
            )
            overlord.move(hg_spot, queue=True)


#_______________________________________________________________________________________________________________________
#          ON STEP
#_______________________________________________________________________________________________________________________

    def _sync_tech_spawn_inhibitors(self) -> None:
        has_any_lair = self.townhalls.of_type({UnitID.LAIR}).amount > 0
        has_any_hive = self.townhalls.of_type({UnitID.HIVE}).amount > 0
        morphing_lair = any(
            th.type_id == UnitID.HATCHERY
            and any(order.ability.id == AbilityId.UPGRADETOLAIR_LAIR for order in th.orders)
            for th in self.townhalls
        )
        morphing_hive = any(
            any(order.ability.id == AbilityId.UPGRADETOHIVE_HIVE for order in th.orders)
            for th in self.townhalls
        )
        pending_lair = self.already_pending(UnitID.LAIR) > 0 or morphing_lair
        pending_hive = self.already_pending(UnitID.HIVE) > 0 or morphing_hive

        # Keep these inhibitors in sync even while tech morphs are in progress.
        if has_any_hive or pending_hive:
            self.spawn_inhibitors.discard("building_hive")
            self.spawn_inhibitors.discard("building_lair")
        elif has_any_lair or pending_lair:
            self.spawn_inhibitors.discard("building_lair")

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)

        # Some game starts can call on_start before townhalls are populated.
        # Recover safely on the first step where a townhall is visible.
        if self.first_base is None and self.townhalls:
            self.first_base = self.townhalls.first
        if self.first_base is None:
            return

        self._sync_tech_spawn_inhibitors()

        await self.debug_tool()



        self._macro()


        # https://aressc2.github.io/ares-sc2/api_reference/manager_mediator.html#ares.managers.manager_mediator.ManagerMediator.get_units_from_role
        # see `self.on_unit_created` where we originally assigned units ATTACKING role //
        forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        # Separate air attackers from ground forces.
        # Corruptors and mutalisks always intercept air threats independently of _commenced_attack.
        _AIR_ATTACKER_TYPES: frozenset = frozenset({UnitID.CORRUPTOR, UnitID.MUTALISK})
        air_forces: Units = forces.filter(lambda u: u.type_id in _AIR_ATTACKER_TYPES)

        if self._commenced_attack:
            self._micro(forces)

        elif self.get_total_supply(forces) >= self._begin_attack_at_supply:
            self._commenced_attack = True

        # Even when the main army is not yet attacking, air interceptors react
        # to any enemy flying unit over our creep (e.g. Battlecruiser, Banshee).
        if not self._commenced_attack and air_forces:
            enemy_air_on_creep: Units = self.enemy_units.filter(
                lambda u: u.is_flying and not u.is_memory and self.has_creep(u.position)
            )
            if enemy_air_on_creep:
                self._micro(air_forces)

        # Fallback tosco: tenta lançar bile a cada frame, além do controle no _micro.
        await self._force_ravager_bile_each_frame()



        await self.return_to_base(forces)
        await self.retreat_3rd_and_4rd_overlord()



        if self.EnemyRace == Race.Terran:
            #await self.build_queens()
            # --- Vital (every frame) ---
            await self.attack_reaper()
            await self.attack_banshee()
            await self.defend()
            await self.turnOffSpawningControllerOnEarlyGame()

            # --- % 4 == 0: detection / scouting / burrow ---
            if iteration % 4 == 0:
                await self.is_terran_agressive()
                await self.is_bunker_rush()
                await self.search_proxy_barracks()
                await self.findReaper()
                await self.burrow_roaches()
                await self.burrow_infestors()
                await self.scout_enemy_base_with_changeling()
                await self.is_marine_rush()

            # --- % 4 == 1: building / construction ---
            elif iteration % 4 == 1:
                await self.is_structures_flying()
                await self.find_liberator()
                await self.is_3_base_terran()
                await self.is_worker_rush()
                #await self.build_hydra_den()
                await self.force_complete_build_order()
                await self.create_queens_after_build_order()
                await self.build_evolution_chamber()

            # --- % 4 == 2: composition checks ---
            elif iteration % 4 == 2:
                await self.is_mass_marauder()
                await self.is_mass_liberator()
                await self.make_ravagers()
                await self.is_mass_widow_mine()

            # --- % 4 == 3: upgrades / mid-game checks ---
            elif iteration % 4 == 3:
                await self.is_mid_game()
                await self.make_roach_speed()
                # await self.use_fungal_growth()
                # await self.throw_bile()
                await self.is_bc()
                await self.is_mass_tank()
                await self.is_late_game_vs_terran()
                await self.build_missle_upgrades()

            if "Worker_Rush" in self.enemy_strategy:
                await self.change_to_bo_TwelvePool()
                await self.build_roach_warren()
                await self.defend_worker_rush()    


            if "Bunker_Rush" in self.enemy_strategy:
                if iteration % 4 == 0:
                    await self.cancel_second_base()
                elif iteration % 4 == 1:
                    await self.build_roach_warren()
                elif iteration % 4 == 2:
                    await self.research_burrow()
                elif iteration % 4 == 3:
                    await self.change_to_bo_Bunker_Rush()
            #if "2_Base_Terran" in self.enemy_strategy:


            if "Proxy_Barracks" in self.enemy_strategy:
                #await self.cancel_second_base()
                # Vital: active micro every frame
                await self.retreat_overlords()
                await self.harass_worker_proxy_barracks()
                if iteration % 4 == 1:
                    await self.build_spine_crawlers()
                elif iteration % 4 == 2:
                    await self.change_to_bo_DefensiveVsProxyBarracks()


            if "Banshee" in self.enemy_strategy:
                if iteration % 4 == 0:
                    await self.make_spores()
                elif iteration % 4 == 1:
                    await self.make_overseer()
                elif iteration % 4 == 2:
                    await self.is_mass_banshee()
                if iteration % 8 == 0:
                    await self.make_changeling()
                    await self.move_changeling()
                    await self.assign_overseer()


            if "Liberator" in self.enemy_strategy:
                if iteration % 4 == 0:
                    await self.make_spores()


            if "Flying_Structures" in self.enemy_strategy:
                #await self.build_lair()
                #await self.build_hydra_den()
                #self.register_behavior(BuildWorkers(to_count=80))
                if iteration % 4 == 1:
                    await self.build_spire()
                elif iteration % 4 == 2:
                    #await self.build_second_gas()
                    await self.build_four_gas()
                    await self.build_more_queens()
                if iteration % 32 == 0:
                    await self.spread_overlords()


            if "Mass_Banshee" in self.enemy_strategy:
                #await self.build_lair()
                #await self.build_hydra_den()
                #self.register_behavior(BuildWorkers(to_count=80))
                if iteration % 4 == 1:
                    await self.build_spire()
                elif iteration % 4 == 2:
                    #await self.build_second_gas()
                    await self.build_four_gas()
                    await self.build_more_queens()

            if "Terran_Agressive" in self.enemy_strategy:
                #await self.build_roach_warren()
                if iteration % 4 == 1:
                    await self.build_spine_crawlers()
                elif iteration % 4 == 2:
                    await self.change_to_bo_Terran_Agressive()



            if "Mass_Widow_Mine" in self.enemy_strategy:
                if iteration % 4 == 0:
                    await self.make_overseer()
                if iteration % 8 == 0:
                    await self.assign_overseer()
                    await self.make_changeling()
                    await self.move_changeling()

            if "Mid_Game" in self.enemy_strategy:
                # Vital: army/protocol management every frame
                await self.mid_game_protocol()
                if iteration % 4 == 0:
                    await self.build_infestation_pit()
                elif iteration % 4 == 1:
                    await self.build_lair()
                elif iteration % 4 == 2:
                    await self.build_evolution_chamber()


            if "Battlecruiser" in self.enemy_strategy:
                if iteration % 4 == 0:
                    await self.build_spire()
                elif iteration % 4 == 1:
                    await self.build_spores_vs_bc()
                elif iteration % 4 == 2:
                    await self.build_more_queens()


            if "2_Base_Terran" in self.enemy_strategy:
                pass  # infestation pit já é ordenado via "Mid_Game"


            if "Late_Game" in self.enemy_strategy:
                await self.late_game_vs_terran_protocol()
                if iteration % 4 == 1:
                    await self.build_hive()
                    await self.make_one_viper()


        if self.EnemyRace == Race.Protoss:
            await self.build_queens()
            await self.is_protoss_agressive()
            await self.build_mellee_upgrades()
            await self.build_armor_upgrades()
            if iteration % 4 == 0:
                await self.burrow_roaches()
            await self.defend()
            await self.search_proxy_vs_protoss()
            await self.is_proxy_stargate()
            await self.is_worker_rush()
            await self.is_mid_game_vs_protoss()
            await self.is_cannon_rush()
            await self.make_ravagers()
            await self.is_worker_rush()
            await self.check_invisible_units()

            if "Proxy_Stargate" in self.enemy_strategy:
                self._apply_proxy_stargate_queen_policy()

            if "Proxy_Stargate" in self.enemy_strategy:
                await self.build_spores_vs_proxy_stargate()
                await self.stop_collecting_gas()

            if "Worker_Rush" in self.enemy_strategy:
                await self.change_to_bo_TwelvePool()
                await self.build_roach_warren()
                await self.defend_worker_rush()    

            if "Protoss_Agressive" in self.enemy_strategy:
                #await self.build_spine_crawlers()
                if "Cannon_Rush" not in self.enemy_strategy:
                    await self.build_2_spine_crawlers()
                await self.change_to_bo_Protoss_Agressive()

            if "2_Base_Protoss" in self.enemy_strategy and "Cannon_Rush" not in self.enemy_strategy:
                if not self.mid_game:
                    await self.stop_collecting_gas()

            if "2_Proxy_Gateway" in self.enemy_strategy:
                await self.cancel_second_base()
                await self.retreat_overlords()
                await self.make_spines_on_main()
                await self.build_roach_warren()
                await self.research_burrow()
                #await self.make_macro_hatch()

            if "Mid_Game" in self.enemy_strategy:
                await self.mid_game_vs_protoss_protocol()
                await self.build_lair()


            if "Cannon_Rush" in self.enemy_strategy:
                await self.cancel_second_base()
                await self.research_burrow()
                await self.change_to_bo_CannonRush()
                await self.make_macro_hatch()
                await self.emergency_supply_block()
                await self.worker_attack_cannon_rush()
                self._begin_attack_at_supply = 40


        if self.EnemyRace == Race.Zerg:
            if iteration % 8 == 0:
                await self.assign_overseer()
            await self.find_cheese_spine_crawler()
            if iteration % 4 == 0:
                await self.burrow_roaches()
            await self.find_mutalisks()
            await self.is_worker_rush()
            await self.force_complete_build_order()
            #await self.zergling_scout()
            await self.make_overseer()
            await self.turnOffSpawningControllerOnEarlyGame()
            #await self.build_one_spine_crawler()
            if iteration % 8 == 0:
                await self.make_changeling()
                await self.move_changeling()
            await self.is_ling_rush()
            await self.is_twelve_pool()
            await self.build_roach_warren_failed()
            await self.check_nydus_spot()
            await self.build_missle_upgrades()
            await self.make_ravagers()
            await self.is_worker_rush()
            


            if not any(tag in self.enemy_strategy for tag in ("Worker_Rush", "Ling_Rush", "12_Pool")):
                await self.build_one_spine_crawler()

            if "Worker_Rush" in self.enemy_strategy:
                await self.change_to_bo_TwelvePool()
                await self.build_roach_warren()
                await self.defend_worker_rush()    

            if "Mutalisk" in self.enemy_strategy:
                await self.make_spores()
        
            #if "Cheese_Spine_Crawler" in self.enemy_strategy:
                #await self.turnOffSpeedMining()


            if "Ling_Rush" in self.enemy_strategy or "12_Pool" in self.enemy_strategy:
                await self.build_roach_warren()
                #await self.make_spines_on_main()
                await self.change_to_bo_Vs_Ling_Rush()
                await self.cancel_second_base()
                await self.make_spines_vs_ling_rush()



        if self.EnemyRace == Race.Random:
            await self.build_queens()
            await self.discover_race()
            if iteration % 4 == 0:
                await self.burrow_roaches()
            await self.defend()
            await self.is_worker_rush()

            if "Worker_Rush" in self.enemy_strategy:
                await self.change_to_bo_TwelvePool()
                await self.build_roach_warren()
                await self.defend_worker_rush()    

            if "Random_Protoss" in self.enemy_strategy:
                await self.build_queens()
                await self.is_protoss_agressive()
                if iteration % 4 == 0:
                    await self.burrow_roaches()
                await self.defend()
                await self.search_proxy_vs_protoss()
                await self.is_proxy_stargate()
                await self.is_worker_rush()
                await self.is_mid_game_vs_protoss()
                await self.is_cannon_rush()
                await self.make_ravagers()
                await self.check_invisible_units()

                if "Proxy_Stargate" in self.enemy_strategy:
                    self._apply_proxy_stargate_queen_policy()

                if "Proxy_Stargate" in self.enemy_strategy:
                    await self.build_spores_vs_proxy_stargate()
                    await self.stop_collecting_gas()

                if "2_Base_Protoss" in self.enemy_strategy and "Cannon_Rush" not in self.enemy_strategy:
                    await self.stop_collecting_gas()

                if "Mid_Game" in self.enemy_strategy:
                    await self.mid_game_vs_protoss_protocol()
                    await self.build_lair()

                if "Cannon_Rush" in self.enemy_strategy:
                    await self.cancel_second_base()
                    await self.research_burrow()
                    await self.change_to_bo_CannonRush()
                    await self.make_macro_hatch()
                    await self.emergency_supply_block()
                    await self.worker_attack_cannon_rush()
                    self._begin_attack_at_supply = 40

                else:
                    if "Protoss_Agressive" in self.enemy_strategy:  
                        #await self.change_to_bo_VsOneBaseRandomProtoss()
                        await self.build_spine_crawlers()
                        await self.change_to_bo_Protoss_Agressive()
                        self._begin_attack_at_supply = 40

            if "Random_Terran" in self.enemy_strategy:
                # --- Vital (every frame) ---
                await self.attack_reaper()
                await self.attack_banshee()
                await self.defend()
                await self.turnOffSpawningControllerOnEarlyGame()
                await self.scout_enemy_base_with_changeling()

                # --- % 4 == 0: detection / scouting / burrow ---
                if iteration % 4 == 0:
                    await self.is_terran_agressive()
                    await self.is_bunker_rush()
                    await self.search_proxy_barracks()
                    await self.findReaper()
                    await self.burrow_roaches()
                    await self.burrow_infestors()

                # --- % 4 == 1: building / construction ---
                elif iteration % 4 == 1:
                    await self.is_structures_flying()
                    await self.find_liberator()
                    await self.is_3_base_terran()
                    await self.is_worker_rush()
                    #await self.build_hydra_den()
                    await self.force_complete_build_order()
                    await self.create_queens_after_build_order()

                # --- % 4 == 2: composition checks ---
                elif iteration % 4 == 2:
                    await self.is_mass_marauder()
                    await self.is_mass_liberator()
                    await self.make_ravagers()
                    await self.is_mass_widow_mine()

                # --- % 4 == 3: upgrades / mid-game checks ---
                elif iteration % 4 == 3:
                    await self.is_mid_game()
                    await self.make_roach_speed()
                    # await self.use_fungal_growth()
                    # await self.throw_bile()
                    await self.is_bc()
                    await self.is_mass_tank()
                    await self.is_late_game_vs_terran()


                if "Bunker_Rush" in self.enemy_strategy:
                    if iteration % 4 == 0:
                        await self.cancel_second_base()
                    elif iteration % 4 == 1:
                        await self.build_roach_warren()
                    elif iteration % 4 == 2:
                        await self.research_burrow()
                    elif iteration % 4 == 3:
                        await self.change_to_bo_Bunker_Rush()
                #if "2_Base_Terran" in self.enemy_strategy:


                if "Proxy_Barracks" in self.enemy_strategy:
                    #await self.cancel_second_base()
                    # Vital: active micro every frame
                    await self.retreat_overlords()
                    await self.harass_worker_proxy_barracks()
                    if iteration % 4 == 1:
                        await self.build_spine_crawlers()
                    elif iteration % 4 == 2:
                        await self.change_to_bo_DefensiveVsProxyBarracks()


                if "Banshee" in self.enemy_strategy:
                    if iteration % 4 == 0:
                        await self.make_spores()
                    elif iteration % 4 == 1:
                        await self.make_overseer()
                    elif iteration % 4 == 2:
                        await self.is_mass_banshee()
                    if iteration % 8 == 0:
                        await self.make_changeling()
                        await self.move_changeling()
                        await self.assign_overseer()


                if "Liberator" in self.enemy_strategy:
                    if iteration % 4 == 0:
                        await self.make_spores()


                if "Flying_Structures" in self.enemy_strategy:
                    #await self.build_lair()
                    #await self.build_hydra_den()
                    #self.register_behavior(BuildWorkers(to_count=80))
                    if iteration % 4 == 1:
                        await self.build_spire()
                    elif iteration % 4 == 2:
                        #await self.build_second_gas()
                        await self.build_four_gas()
                    if iteration % 32 == 0:
                        await self.spread_overlords()


                if "Mass_Banshee" in self.enemy_strategy:
                    #await self.build_lair()
                    #await self.build_hydra_den()
                    #self.register_behavior(BuildWorkers(to_count=80))
                    if iteration % 4 == 1:
                        await self.build_spire()
                    elif iteration % 4 == 2:
                        #await self.build_second_gas()
                        await self.build_four_gas()

                if "Terran_Agressive" in self.enemy_strategy:
                    #await self.build_roach_warren()
                    if iteration % 4 == 1:
                        await self.build_spine_crawlers()
                    elif iteration % 4 == 2:
                        await self.change_to_bo_Terran_Agressive()

                if "Mass_Widow_Mine" in self.enemy_strategy:
                    if iteration % 4 == 0:
                        await self.make_overseer()
                    if iteration % 8 == 0:
                        await self.assign_overseer()
                        await self.make_changeling()
                        await self.move_changeling()

                if "Mid_Game" in self.enemy_strategy:
                    # Vital: army/protocol management every frame
                    await self.mid_game_protocol()
                    if iteration % 4 == 0:
                        await self.build_infestation_pit()
                    elif iteration % 4 == 1:
                        await self.build_lair()
                    elif iteration % 4 == 2:
                        await self.build_evolution_chamber()
                    elif iteration % 4 == 3:
                        await self.build_missle_upgrades()

                if "Battlecruiser" in self.enemy_strategy:
                    if iteration % 4 == 0:
                        await self.build_spire()
                    elif iteration % 4 == 1:
                        await self.build_spores_vs_bc()
                    elif iteration % 4 == 2:
                        await self.build_more_queens()


            if "Random_Zerg" in self.enemy_strategy:
                await self.is_twelve_pool()
                if iteration % 8 == 0:
                    await self.assign_overseer()
                await self.find_cheese_spine_crawler()
                if iteration % 4 == 0:
                    await self.burrow_roaches()
                await self.find_mutalisks()
                await self.is_worker_rush()
                await self.force_complete_build_order()
                #await self.zergling_scout()
                await self.make_overseer()
                await self.turnOffSpawningControllerOnEarlyGame()
                #await self.build_one_spine_crawler()
                if iteration % 8 == 0:
                    await self.make_changeling()
                    await self.move_changeling()
                await self.is_ling_rush()
                await self.build_roach_warren_failed()
                await self.check_nydus_spot()
                await self.build_missle_upgrades()
                await self.make_ravagers()

                if "Ling_Rush" in self.enemy_strategy or "12_Pool" in self.enemy_strategy:
                    await self.build_roach_warren()
                    #await self.make_spines_on_main()
                    await self.change_to_bo_Vs_Ling_Rush()
                    await self.cancel_second_base()
                    await self.make_spines_vs_ling_rush()



#_______________________________________________________________________________________________________________________
#          QUEENS
#_______________________________________________________________________________________________________________________

        queens: Units = self.units(UnitID.QUEEN)
        proxy_stargate_mode = "Proxy_Stargate" in self.enemy_strategy

        # Se supply > 195, queens largam tudo e atacam qualquer inimigo visível
        if self.supply_used > 195 and not proxy_stargate_mode:
            visible_enemies: Units = self.enemy_units.filter(lambda u: not u.is_memory)
            if visible_enemies:
                if iteration % 8 == 0:
                    closest_enemy = cy_closest_to(self.start_location, visible_enemies)
                    if closest_enemy.tag != self._queen_attack_target:
                        self._queen_attack_target = closest_enemy.tag
                        for queen in queens:
                            queen.attack(closest_enemy.position)
            else:
                self._queen_attack_target = None
        else:
            # A biblioteca gerencia tudo: 1 inject queen por base (priority=1)
            # e o restante espalha creep (priority=0, max=20)
            await self.queens.manage_queens(iteration, queens)
            await self._hold_queens_near_hatcheries_vs_proxy_stargate(queens, iteration)

            # Se algum inimigo pisar na gosma, reavalia o alvo a cada 8 iterações para não inundar o motor com ordens
            if iteration % 8 == 0 and not proxy_stargate_mode:
                enemy_on_creep: Units = self.enemy_units.filter(
                    lambda u: not u.is_memory and self.has_creep(u.position)
                )
                if enemy_on_creep:
                    closest_enemy = cy_closest_to(self.start_location, enemy_on_creep)
                    if closest_enemy.tag != self._queen_attack_target:
                        self._queen_attack_target = closest_enemy.tag
                        for queen in queens:
                            if queen.energy < 25:  # não está injetando (energia < custo do inject)
                                queen.attack(closest_enemy.position)
                else:
                    self._queen_attack_target = None



    async def return_to_base(self, forces: Units) -> None:
        # Helper: escolhe a base de referência
        def _get_base_ref():
            bases = self.structures(UnitID.HATCHERY).ready
            if bases.amount >= 2 and self.second_base is not None:
                return self.second_base
            return self.first_base

        if self._commenced_attack:
            # If we don't have enough army, stop attacking and build more units

            # RETURN TO BASE
            if self.get_total_supply(forces) < 0.5 * self._begin_attack_at_supply:
                base_ref = _get_base_ref()

                # Verifica se há inimigos próximos da base de referência ou na creep
                base_under_attack = False
                for enemy in self.enemy_units:
                    if enemy.distance_to(base_ref.position) < 18 or self.has_creep(enemy.position):
                        base_under_attack = True
                        break

                if base_under_attack:
                    self._retreating = False  # cancela retreat: base precisa ser defendida
                    # Mantém o modo ataque
                else:
                    self._commenced_attack = False
                    self.is_roach_attacking = False
                    self._retreating = True
                    retreat_pos = base_ref.position.towards(self.game_info.map_center, 6)
                    for unit in forces:
                        unit.move(retreat_pos)

        # Enquanto retirando, re-envia ordem a cada ~32 frames para unidades ainda longe da base
        elif self._retreating:
            base_ref = _get_base_ref()
            retreat_pos = base_ref.position.towards(self.game_info.map_center, 6)

            # Verifica se algum inimigo invadiu durante a retirada — retoma o ataque
            for enemy in self.enemy_units:
                if enemy.distance_to(base_ref.position) < 18 or self.has_creep(enemy.position):
                    self._retreating = False
                    self._commenced_attack = True
                    return

            stragglers = [u for u in forces if u.distance_to(retreat_pos) > 12]
            if stragglers:
                # Throttle: reemite ordem a cada ~32 game loops (~1.4 s em realtime)
                if self.state.game_loop % 32 == 0:
                    for unit in stragglers:
                        unit.move(retreat_pos)
            else:
                self._retreating = False

    async def build_queens(self):
        for th in self.townhalls.ready:
            # Check if the number of queens is less than the number of townhalls
            if len(self.units(UnitID.QUEEN)) <= len(self.townhalls.ready) + 1:
                # Check if we're not already training a queen
                if not self.already_pending(UnitID.QUEEN):
                    # If we're not, train a queen
                    self.do(th.train(UnitID.QUEEN))



    async def build_mellee_upgrades(self):
        if self.structures(UnitID.EVOLUTIONCHAMBER).ready:
            if self.structures(UnitID.SPAWNINGPOOL).ready:
                if not self.already_pending_upgrade(UpgradeId.ZERGLINGATTACKSPEED):
                    if self.can_afford(UpgradeId.ZERGLINGATTACKSPEED):
                        self.research(UpgradeId.ZERGLINGATTACKSPEED)
                if not self.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL1):
                    if self.can_afford(UpgradeId.ZERGMELEEWEAPONSLEVEL1):
                        self.research(UpgradeId.ZERGMELEEWEAPONSLEVEL1)
                if not self.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL2):
                    if self.can_afford(UpgradeId.ZERGMELEEWEAPONSLEVEL2):
                        self.research(UpgradeId.ZERGMELEEWEAPONSLEVEL2)
                if not self.already_pending_upgrade(UpgradeId.ZERGMELEEWEAPONSLEVEL3):
                    if self.can_afford(UpgradeId.ZERGMELEEWEAPONSLEVEL3):
                        self.research(UpgradeId.ZERGMELEEWEAPONSLEVEL3)

    async def build_armor_upgrades(self):
        if self.structures(UnitID.EVOLUTIONCHAMBER).ready:
            if self.structures(UnitID.SPAWNINGPOOL).ready:
                if not self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                    if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                        self.research(UpgradeId.ZERGGROUNDARMORSLEVEL1)
                if not self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL2):
                    if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL2):
                        self.research(UpgradeId.ZERGGROUNDARMORSLEVEL2)
                if not self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL3):
                    if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL3):
                        self.research(UpgradeId.ZERGGROUNDARMORSLEVEL3)


    async def build_range_upgrades(self):
        if self.structures(UnitID.EVOLUTIONCHAMBER).ready:
            if self.structures(UnitID.SPAWNINGPOOL).ready:
                if not self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL1):
                    if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL1):
                        self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL1)
                if not self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL2):
                    if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL2):
                        self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL2)
                if not self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL3):
                    if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL3):
                        self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL3)



    async def build_lair(self):
        has_lair_tech = bool(self.townhalls.of_type({UnitID.LAIR, UnitID.HIVE}))
        morphing_lair = any(
            th.type_id == UnitID.HATCHERY
            and any(order.ability.id == AbilityId.UPGRADETOLAIR_LAIR for order in th.orders)
            for th in self.townhalls
        )
        morphing_hive = any(
            any(order.ability.id == AbilityId.UPGRADETOHIVE_HIVE for order in th.orders)
            for th in self.townhalls
        )
        pending_lair = self.already_pending(UnitID.LAIR) > 0 or morphing_lair
        pending_hive = self.already_pending(UnitID.HIVE) > 0 or morphing_hive

        # Hive replaces Lair as townhall tech; a pending morph also satisfies this.
        if has_lair_tech or pending_lair or pending_hive:
            self.spawn_inhibitors.discard("building_lair")
            return

        if not self.structures(UnitID.LAIR) and not self.already_pending(UnitID.LAIR):
            self.spawn_inhibitors.add("building_lair")
            if self.can_afford(UnitID.LAIR):
                th: Unit = self.first_base
                th(AbilityId.UPGRADETOLAIR_LAIR)
        else:
            self.spawn_inhibitors.discard("building_lair")



    async def build_hydra_den(self):
        if self.structures(UnitID.LAIR).ready:
            if self.structures(UnitID.HYDRALISKDEN).amount == 0 and not self.already_pending(UnitID.HYDRALISKDEN):
                if self.tag_worker_build_hydra_den == 0:
                    if self.can_afford(UnitID.HYDRALISKDEN):
                        positions = self.mediator.get_behind_mineral_positions(th_pos=self.first_base.position)
                        reference = positions[1] if positions else None
                        target = reference.towards(self.first_base, -1)


                        #await self.build(UnitID.HYDRALISKDEN, near=target)
                        if worker := self.mediator.select_worker(target_position=target):                
                            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                            self.tag_worker_build_hydra_den = worker
                            #self.mediator.build_with_specific_worker(worker, UnitID.HATCHERY, target, BuildingPurpose.NORMAL_BUILDING)
                            self.mediator.build_with_specific_worker(worker=self.tag_worker_build_hydra_den, structure_type=UnitID.HYDRALISKDEN, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)

    async def discover_race(self):
        if self.random_race_discovered == False:
            if self.time < 60:
                _TERRAN = frozenset({'CommandCenter', 'SupplyDepot', 'Barracks', 'SCV'})
                _ZERG   = frozenset({'Hatchery', 'SpawningPool', 'Drone', 'Overlord'})
                _PROTOSS = frozenset({'Nexus', 'Gateway', 'Pylon', 'Probe'})

                for unit in list(self.enemy_units) + list(self.enemy_structures):
                    if unit.name in _TERRAN:
                        await self.chat_send("Tag: Random_Terran")
                        self.enemy_strategy.append("Random_Terran")
                        self.random_race_discovered = True
                        break
                    elif unit.name in _ZERG:
                        await self.chat_send("Tag: Random_Zerg")
                        self.enemy_strategy.append("Random_Zerg")
                        self.random_race_discovered = True
                        break
                    elif unit.name in _PROTOSS:
                        await self.chat_send("Tag: Random_Protoss")
                        self.enemy_strategy.append("Random_Protoss")
                        self.random_race_discovered = True
                        break


    async def build_spine_crawlers(self):
        if not self.rally_point_set:
            return

        # Base de referência: segunda base se existir, senão a primeira
        base = getattr(self, "second_base", None) or self.first_base
        base_pos = base.position

        # Vetor da base em direção ao centro do mapa (frente)
        to_center = self.game_info.map_center - base_pos
        mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
        dir_unit = Point2((to_center.x / mag, to_center.y / mag))

        # Vetor perpendicular (eixo da linha dos spines)
        line_unit = Point2((-dir_unit.y, dir_unit.x))

        # Parâmetros da linha
        forward_offset = 6.0   # quão à frente da base
        spacing = 2.5          # distância lateral entre spines

        # Âncora (ponto central da linha)
        anchor = base_pos.towards(self.game_info.map_center, forward_offset)

        # Slots-alvo em linha: esquerda, centro, direita (ordem de construção: centro, esquerda, direita)
        # Empurra cada slot +1.0 na direção de construção (afastando da segunda base)
        # Slots-alvo em linha: esquerda, centro, direita (ordem de construção: centro, esquerda, direita)
        slot_center = anchor
        slot_left   = Point2((anchor.x - line_unit.x * spacing, anchor.y - line_unit.y * spacing))
        slot_right  = Point2((anchor.x + line_unit.x * spacing, anchor.y + line_unit.y * spacing))
        slots = [slot_center, slot_left, slot_right]

        def has_spine_near(p: Point2, radius: float = 2.0) -> bool:
            # Considera spines prontos e em construção
            return any(s.distance_to(p) <= radius for s in self.structures(UnitID.SPINECRAWLER))

        async def place_spine_at(pos: Point2) -> Point2 | None:
            # Se não houver creep exato, tenta pequenos ajustes laterais na própria linha
            candidate = pos
            if not self.has_creep(candidate):
                found = None
                for d in (0.5, 1.0, 1.5, 2.0):
                    for sign in (1, -1):
                        test = Point2((pos.x + line_unit.x * d * sign, pos.y + line_unit.y * d * sign))
                        if self.has_creep(test):
                            found = test
                            break
                    if found:
                        break
                if found:
                    candidate = found
                else:
                    # último recurso: recua levemente em direção à base para pegar creep
                    candidate = pos.towards(base_pos, 1.0)

            # Refina com find_placement para evitar colisão/minérios
            try:
                placed = await self.find_placement(UnitID.SPINECRAWLER, near=candidate, placement_step=1)
                # Se achou posição, garantir espaçamento mínimo de 2.0 contra spines existentes/em construção
                if placed is not None:
                    min_spacing = 2.0
                    def spaced_ok(p: Point2) -> bool:
                        return all(p.distance_to(s.position) >= min_spacing for s in self.structures(UnitID.SPINECRAWLER))

                    if spaced_ok(placed):
                        return placed

                    # Tenta deslocar lateralmente ao longo da linha para ganhar espaço
                    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                        for sign in (1, -1):
                            shifted = Point2((placed.x + line_unit.x * d * sign, placed.y + line_unit.y * d * sign))
                            if not self.has_creep(shifted):
                                continue
                            try:
                                alt = await self.find_placement(UnitID.SPINECRAWLER, near=shifted, placement_step=1)
                            except Exception:
                                alt = None
                            if alt is not None and spaced_ok(alt):
                                return alt
            except Exception:
                pass
            # fallback: retorna candidato somente se houver creep e espaçamento adequado
            if self.has_creep(candidate):
                min_spacing = 2.0
                if all(candidate.distance_to(s.position) >= min_spacing for s in self.structures(UnitID.SPINECRAWLER)):
                    return candidate
            return None

        # Mapeia cada slot para o atributo de tag do seu worker
        slot_attr = {
            0: "tag_worker_build_spine_crawler",      # centro
            1: "tag_worker_build_2nd_spine_crawler",  # esquerda
            2: "tag_worker_build_3rd_spine_crawler",  # direita
        }

        # Constrói no máximo 3 spines, mantendo alinhamento em linha
        for idx, pos in enumerate(slots):
            # Ordem sequencial: só inicia o 2º após o 1º começar,
            # e só inicia o 3º após o 2º começar.
            if idx == 1 and not (has_spine_near(slots[0]) or getattr(self, slot_attr[0]) != 0):
                break
            if idx == 2 and not (has_spine_near(slots[1]) or getattr(self, slot_attr[1]) != 0):
                break
            # Pula se já existe um spine nesse slot
            if has_spine_near(pos):
                continue
            # Pula se já temos worker associado a esse slot
            if getattr(self, slot_attr[idx]) != 0:
                continue
            # Verifica recursos
            if not self.can_afford(UnitID.SPINECRAWLER):
                break

            build_pos = await place_spine_at(pos)
            if not build_pos:
                continue

            if worker := self.mediator.select_worker(target_position=build_pos):
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                setattr(self, slot_attr[idx], worker)
                self.mediator.build_with_specific_worker(
                    worker=getattr(self, slot_attr[idx]),
                    structure_type=UnitID.SPINECRAWLER,
                    pos=build_pos,
                    building_purpose=BuildingPurpose.NORMAL_BUILDING,
                )
                # print opcional:
                # print(f"Spine Crawler slot {idx} at {build_pos}")

    async def is_terran_agressive(self):
        if "2_Base_Terran" not in self.enemy_strategy and "Terran_Agressive" not in self.enemy_strategy:
            #verify if the terran opponent has only one base. If so, it is an agressive terran and build a spine crawler
            if self.time > 170 and self.time < 180:
                found_cc = False
                for unit in self.enemy_structures:
                    if unit.name == 'CommandCenter' or unit.name == 'OrbitalCommand' or unit.name == 'PlanetaryFortress':
                        if unit.distance_to(self.mediator.get_enemy_nat) < 5:
                            found_cc = True
                            break  # Breake the loop if find the Command Center
                if not found_cc:
                    await self.chat_send("Tag: Terran_Agressive")
                    self.enemy_strategy.append("Terran_Agressive")
                else:
                    await self.chat_send("Tag: 2_Base_Terran")
                    self.enemy_strategy.append("2_Base_Terran")

                # In either case, move the first overlord back to own base
                self.first_overlord.move(self.start_location)



    async def is_protoss_agressive(self):
        if "2_Base_Protoss" not in self.enemy_strategy and "Protoss_Agressive" not in self.enemy_strategy:
        #verify if the protoss opponent has only one base. If so, it is an agressive terran and build a spine crawler
            if self.time > 142 and self.time < 143:
                found_nexus = False
                for unit in self.enemy_structures:
                    if unit.name == 'Nexus':
                        if unit.distance_to(self.mediator.get_enemy_nat) <3 :
                            found_nexus = True
                            break  # Breake the loop if find the Nexus
                if not found_nexus:
                    await self.chat_send("Tag: Protoss_Agressive")
                    self.enemy_strategy.append("Protoss_Agressive")
                else:
                    await self.chat_send("Tag: 2_Base_Protoss")
                    self.enemy_strategy.append("2_Base_Protoss")



    async def is_bunker_rush(self):
        if not "Bunker_Rush" in self.enemy_strategy:
        #verify if the protoss opponent has only one base. If so, it is an agressive terran and build a spine crawler
            if self.time > 100 and self.time < 102:
                found_bunker = False
                for unit in self.enemy_structures:
                    if unit.name == 'Bunker':
                        if unit.distance_to(self.mediator.get_enemy_nat) > 20:
                            found_bunker = True
                            break  
                if found_bunker:
                    await self.chat_send("Tag: Bunker_Rush")
                    self.enemy_strategy.append("Bunker_Rush")


    async def build_roach_warren(self):
        if self.structures(UnitID.SPAWNINGPOOL).ready:
            if self.structures(UnitID.ROACHWARREN).amount == 0 and not self.already_pending(UnitID.ROACHWARREN):
                if self.tag_worker_build_roach_warren == 0:
                    if self.can_afford(UnitID.ROACHWARREN):
                        map_center = self.game_info.map_center
                        position_towards_map_center = self.start_location.towards(map_center, distance=5)
                        target = await self.find_placement(UnitID.ROACHWARREN, near=position_towards_map_center, placement_step=1)
                        #await self.build(UnitID.HYDRALISKDEN, near=target)
                        if worker := self.mediator.select_worker(target_position=target):                
                            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                            self.tag_worker_build_roach_warren = worker
                            #self.mediator.build_with_specific_worker(worker, UnitID.HATCHERY, target, BuildingPurpose.NORMAL_BUILDING)
                            self.mediator.build_with_specific_worker(worker=self.tag_worker_build_roach_warren, structure_type=UnitID.ROACHWARREN, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)

    async def research_burrow(self):
        if self.structures(UnitID.ROACHWARREN).ready:
            if not self.already_pending_upgrade(UpgradeId.BURROW):
                if self.can_afford(UpgradeId.BURROW):
                    self.research(UpgradeId.BURROW)



    async def search_proxy_barracks(self):
        if self.time < 94:
            if self.one_proxy_barracks_found == False:
                for unit in self.enemy_structures:
                    if unit.name == 'Barracks':
                        if unit.distance_to(self.mediator.get_enemy_nat) > 30:
                            self.one_proxy_barracks_found = True
                            await self.chat_send("Tag: Proxy_Barracks")
                            self.enemy_strategy.append("Proxy_Barracks")
                            break
    
            if self.two_proxy_barracks_found == False:
                # Filtra todos os barracks a mais de 30 do enemy nat
                proxy_barracks = [
                    structure for structure in self.enemy_structures
                    if structure.name == "Barracks" and structure.distance_to(self.mediator.get_enemy_nat) > 30
                ]
                if len(proxy_barracks) >= 2:
                    await self.chat_send("Tag: 2 Proxy_Barracks")
                    self.enemy_strategy.append("2_Proxy_Barracks")
                    self.two_proxy_barracks_found = True



    async def build_second_gas(self):
        if self.structures(UnitID.HATCHERY).amount == 2:
            if self._gas_count_registered < 2:
                self.register_behavior(GasBuildingController(to_count=2))
                self._gas_count_registered = 2


    async def build_four_gas(self):
        if self.structures(UnitID.HATCHERY).amount >= 2:
            if self._gas_count_registered < 4:
                self.register_behavior(GasBuildingController(to_count=4))
                self._gas_count_registered = 4
            
            


    async def cancel_second_base(self):
        if self.time > 180:
            return
        hatcheries = self.structures(UnitID.HATCHERY)
        if self.second_base_canceled == False:
            if hatcheries:
                for hatchery in hatcheries:
                    if not hatchery.is_ready:
                        # Não cancela se for a macro hatch
                        if self.macro_hatch_pos is not None:
                            if hatchery.position.distance_to(self.macro_hatch_pos) < 5:
                                continue
                        self.mediator.cancel_structure(structure=hatchery)
                        self.second_base_canceled = True


    async def retreat_overlords(self):
        #retreat the overlords to the first base so they don't die
        if self.overlord_retreated == False:
            for overlord in self.units(UnitID.OVERLORD):
                if overlord.distance_to(self.first_base.position) < 30:  # Defina a distância que considera "perto"
                    overlord.move(self.first_base.position)
                    self.overlord_retreated = True



        #if spine_crawler_amount == 0 and self.spineCrawlerCheeseDetected:
            #self.spineCrawlerCheeseDetected = False
            #for drone in self.workers:
                #self.mediator.assign_role(tag = drone.tag, role = UnitRole.GATHERING)
                #self.speedMiningOn = True
                


    async def find_cheese_spine_crawler(self):
        if self.time < 180:
            if self.spineCrawlerCheeseDetected == False:
                for spinecrawler in self.enemy_structures(UnitID.SPINECRAWLER):
                    if spinecrawler.distance_to(self.first_base.position) < 20:
                        if spinecrawler.distance_to(self.mediator.get_enemy_nat) > 30:
                            self.spineCrawlerCheeseDetected = True
                            await self.chat_send("Tag: Cheese Spine Crawler")
                            self.enemy_strategy.append("Cheese_Spine_Crawler")


    async def burrow_roaches(self):
        # Burrow the roaches when they are low health
        for roach in self.units(UnitID.ROACH):
            if roach.health_percentage <= self.BURROW_AT_HEALTH_PERC:
                roach(AbilityId.BURROWDOWN_ROACH)


        for burrowed_roach in self.units(UnitID.ROACHBURROWED):
            if burrowed_roach.health_percentage > self.UNBURROW_AT_HEALTH_PERC:
                burrowed_roach(AbilityId.BURROWUP_ROACH)


    async def findReaper(self):
        if self.reaperFound == False:
            for unit in self.enemy_units:
                if unit.name == 'Reaper':
                    self.reaperFound = True
                    break
            if self.reaperFound:
                await self.chat_send("Tag: Reaper")

    async def attack_reaper(self):
        if self.reaperFound:
            for unit in self.enemy_units:
                if unit.name == 'Reaper':
                    if self.has_creep(unit.position):
                        for queen in self.units(UnitID.QUEEN):
                            if queen.energy < 25:
                                queen.attack(unit.position)

    async def attack_banshee(self):
        if self.bansheeFound == False:
            for unit in self.enemy_units:
                if unit.name == 'Banshee':
                    self.bansheeFound = True
                    break
            if self.bansheeFound:
                await self.chat_send("Tag: Banshee")
                self.enemy_strategy.append("Banshee")

        if self.bansheeFound:
            for unit in self.enemy_units:
                if unit.name == 'Banshee':
                    if self.has_creep(unit.position):
                        for queen in self.units(UnitID.QUEEN):
                            if queen.energy < 25:
                                queen.attack(unit.position)


    async def make_spores(self):
        """Build one Spore Crawler per base. Rebuilds automatically if destroyed."""
        bases = self.townhalls.ready
        if not bases:
            return

        for base in bases:
            # Check if a spore (ready or under construction) already exists near this base
            spore_near = any(
                s.distance_to(base.position) < 10
                for s in self.structures(UnitID.SPORECRAWLER)
            )
            if spore_near:
                # Spore exists — clear worker tracking so we can react if it's destroyed later
                self.spore_workers.pop(base.tag, None)
                # Train 2 extra drones for this base (once only)
                if base.tag not in self._spore_extra_drones_done:
                    drones_trained = 0
                    for larva in self.units(UnitID.LARVA):
                        if drones_trained >= 2:
                            break
                        if self.can_afford(UnitID.DRONE):
                            larva.train(UnitID.DRONE)
                            drones_trained += 1
                    if drones_trained >= 2:
                        self._spore_extra_drones_done.add(base.tag)
                continue

            # No spore near this base — check if we have a worker already heading there
            worker_tag = self.spore_workers.get(base.tag, 0)
            if worker_tag:
                worker_alive = self.units(UnitID.DRONE).find_by_tag(worker_tag)
                if worker_alive is not None:
                    continue  # Worker still on the job
                else:
                    # Worker died — reset so we re-order
                    self.spore_workers[base.tag] = 0

            if not self.can_afford(UnitID.SPORECRAWLER):
                continue

            # Find build position between base and mineral line (close to base)
            positions = self.mediator.get_behind_mineral_positions(th_pos=base.position)
            if positions:
                mineral_ref = positions[0]
                target = base.position.towards(mineral_ref, 3)
                if not self.has_creep(target):
                    target = base.position.towards(mineral_ref, 2)
            else:
                target = base.position.towards(self.game_info.map_center, -4)

            try:
                placed = await self.find_placement(UnitID.SPORECRAWLER, near=target, placement_step=1)
            except Exception:
                placed = None
            target = placed if placed else target

            if not self.has_creep(target):
                continue

            if worker := self.mediator.select_worker(target_position=target):
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                self.spore_workers[base.tag] = worker.tag
                self.mediator.build_with_specific_worker(
                    worker=worker,
                    structure_type=UnitID.SPORECRAWLER,
                    pos=target,
                    building_purpose=BuildingPurpose.NORMAL_BUILDING,
                )


    async def make_spines_on_main(self):
        if not self.structures(UnitID.SPAWNINGPOOL).ready:
            return

        # Calcula posições entre o hatchery e a linha de minerais
        minerals_near_base = self.mineral_field.closer_than(12, self.first_base.position)
        if minerals_near_base:
            mineral_center = minerals_near_base.center
        else:
            # fallback: direção oposta à rampa
            mineral_center = self.first_base.position.towards(self.main_base_ramp.top_center, -8)

        # Vetor da base em direção aos minerais
        dm = mineral_center - self.first_base.position
        mag = (dm.x ** 2 + dm.y ** 2) ** 0.5 or 1.0
        dir_min = Point2((dm.x / mag, dm.y / mag))
        # Vetor perpendicular (linha lateral)
        perp = Point2((-dir_min.y, dir_min.x))

        # Âncora: 3.5 tiles do hatchery em direção aos minerais
        anchor = self.first_base.position.towards(mineral_center, 3.5)
        spacing = 2.5

        # slot1: grudado na base, em direção à segunda base (natural)
        nat = self.mediator.get_own_nat
        slot1 = self.first_base.position.towards(nat, 4)
        slot2 = Point2((anchor.x - perp.x * spacing, anchor.y - perp.y * spacing))  # esquerda
        slot3 = Point2((anchor.x + perp.x * spacing, anchor.y + perp.y * spacing))  # direita

        if self.time < 120:
            if self.structures(UnitID.SPINECRAWLER).amount == 0 and not self.already_pending(UnitID.SPINECRAWLER):
                if self.tag_worker_build_spine_crawler == 0:
                    if self.can_afford(UnitID.SPINECRAWLER):
                        target = slot1
                        if worker := self.mediator.select_worker(target_position=target):
                            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                            self.tag_worker_build_spine_crawler = worker
                            self.mediator.build_with_specific_worker(worker=self.tag_worker_build_spine_crawler, structure_type=UnitID.SPINECRAWLER, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)

        if self.tag_worker_build_spine_crawler != 0:
            if self.can_afford(UnitID.SPINECRAWLER):
                target = slot2
                if not self.has_creep(target):
                    target = slot1
                if worker := self.mediator.select_worker(target_position=target):
                    self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                    self.tag_worker_build_2nd_spine_crawler = worker
                    self.mediator.build_with_specific_worker(
                        worker=self.tag_worker_build_2nd_spine_crawler,
                        structure_type=UnitID.SPINECRAWLER,
                        pos=target,
                        building_purpose=BuildingPurpose.NORMAL_BUILDING
                    )

        if self.tag_worker_build_2nd_spine_crawler != 0:
            if self.can_afford(UnitID.SPINECRAWLER):
                target = slot3
                if not self.has_creep(target):
                    target = slot1
                if worker := self.mediator.select_worker(target_position=target):
                    self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                    self.tag_worker_build_3rd_spine_crawler = worker
                    self.mediator.build_with_specific_worker(worker=self.tag_worker_build_3rd_spine_crawler, structure_type=UnitID.SPINECRAWLER, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)


    async def defend(self):
        enemy_on_creep = False
        air_enemy_on_creep = False
        for enemyUnit in self.enemy_units:
            if self.has_creep(enemyUnit.position):
                if not enemyUnit.is_flying:
                    enemy_on_creep = True
                    self.defending = True
                    self._commenced_attack = True
                    # Adicionar a unidade inimiga ao dicionário enemies_on_creep
                    self.enemies_on_creep[enemyUnit.tag] = enemyUnit
                else:
                    # Unidade voadora sobre a creep (ex: Battlecruiser) — não aciona defesa terrestre
                    air_enemy_on_creep = True
            else:
                # Remover a unidade inimiga do dicionário enemies_on_creep se ela sair da creep
                if enemyUnit.tag in self.enemies_on_creep:
                    del self.enemies_on_creep[enemyUnit.tag]
    
        # Remover unidades inimigas do dicionário se elas não estiverem mais na lista de unidades inimigas
        self.enemies_on_creep = {tag: unit for tag, unit in self.enemies_on_creep.items() if unit in self.enemy_units}
    
        # Só executa o reset se não houver nenhum inimigo (terrestre ou aéreo) sobre a creep.
        # Se houver inimigo aéreo, os corruptors/mutalisks tratam disso no on_step; não resetar aqui.
        if not enemy_on_creep and not air_enemy_on_creep:
            forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
            _AIR_ATTACKER_TYPES = {UnitID.CORRUPTOR, UnitID.MUTALISK}
            ground_forces: Units = forces.filter(lambda u: u.type_id not in _AIR_ATTACKER_TYPES)
            if self.get_total_supply(forces) < self._begin_attack_at_supply:
                self._commenced_attack = False
                self.defending = False
                # Somente envia o comando de recuo uma vez por transição de ataque → recuo.
                if not self._retreat_issued:
                    self._retreat_issued = True
                    for unit in ground_forces:
                        if self.second_base is not None:
                            unit.move(self.second_base.position.towards(self.game_info.map_center, 4))
                        else:
                            unit.move(self.first_base.position.towards(self.game_info.map_center, 6))
            else:
                self._commenced_attack = True
                self._retreat_issued = False  # reseta quando o ataque é retomado


    async def find_mutalisks(self):
        if self.mutalisksFound == False:
            for unit in self.enemy_units:
                if unit.name == 'Mutalisk':
                    self.mutalisksFound = True
                    break
            if self.mutalisksFound:
                await self.chat_send("Tag: Mutalisk")
                self.enemy_strategy.append("Mutalisk")


    async def search_proxy_vs_protoss(self):
        if self.time < 120:
            enemy_main = self.enemy_start_locations[0]
            enemy_natural_location = self.mediator.get_enemy_nat

            def is_proxy(unit) -> bool:
                """Returns True if the structure is far from both enemy main and enemy natural."""
                return (
                    unit.distance_to(enemy_main) > 20
                    and unit.distance_to(enemy_natural_location) > 20
                )

            if self.proxy_pylon_found == False:
                for unit in self.enemy_structures:
                    if unit.name == 'Pylon' and is_proxy(unit):
                        self.proxy_pylon_found = True
                        await self.chat_send("Tag: Proxy_Pylon")
                        self.enemy_strategy.append("Proxy_Pylon")
                        break

            if self.one_proxy_gateWay_found == False:
                for unit in self.enemy_structures:
                    if unit.name == 'Gateway' and is_proxy(unit):
                        self.one_proxy_gateWay_found = True
                        await self.chat_send("Tag: Proxy_Gateway")
                        self.enemy_strategy.append("Proxy_Gateway")
                        break

            if self.two_proxy_gateWay_found == False:
                proxy_gateways_count = sum(
                    1 for structure in self.enemy_structures
                    if structure.name == "Gateway" and is_proxy(structure)
                )
                if proxy_gateways_count > 1:
                    if "Proxy_Gateway" in self.enemy_strategy:
                        await self.chat_send("Tag: 2_Proxy_Gateway")
                        self.enemy_strategy.append("2_Proxy_Gateway")
                        self.two_proxy_gateWay_found = True

            if self.photon_cannon_found == False:
                for unit in self.enemy_structures:
                    if unit.name == 'PhotonCannon':
                        expansion = self.mediator.get_own_nat
                        if unit.distance_to(expansion) < 20:
                            self.photon_cannon_found = True
                            await self.chat_send("Tag: Cannon_Rush")
                            self.enemy_strategy.append("Cannon_Rush")
                            break


    async def is_structures_flying(self):
        # Some terrans lift their structures when they feel they are about to lose.
        # This function aims to recognize this situation to make mutalisks
        if self.time > 240:
            if self.terran_flying_structures == False:
                enemy_main = self.enemy_start_locations[0]
                flying_near_main = sum(
                    1
                    for s in self.enemy_structures
                    if s.is_flying and s.distance_to(enemy_main) < 12
                )
                if flying_near_main >= 2:
                    await self.chat_send("Tag: Flying_Structures")
                    self.enemy_strategy.append("Flying_Structures")
                    self.terran_flying_structures = True

    async def build_spire(self):
        # Manage spawn inhibitor
        spire_exists = self.structures(UnitID.SPIRE) or self.already_pending(UnitID.SPIRE)
        if not spire_exists and self.structures(UnitID.LAIR).ready:
            self.spawn_inhibitors.add("building_spire")
        else:
            self.spawn_inhibitors.discard("building_spire")

        if not self.structures(UnitID.LAIR).ready:
            return

        if spire_exists:
            self.spire_ordered = False
            self._spire_worker_tag = 0
            return

        # If a worker is already on the way, do nothing
        if getattr(self, "_spire_worker_tag", 0):
            worker = self.workers.find_by_tag(self._spire_worker_tag)
            if worker:
                return
            # Worker died — reset so we try again
            self._spire_worker_tag = 0
            self.spire_ordered = False

        if self.spire_ordered:
            return

        if not self.can_afford(UnitID.SPIRE):
            return

        # Try candidate positions around the first base until one is placeable
        base = self.first_base.position
        center = self.game_info.map_center
        to_center = center - base
        mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
        fwd = Point2((to_center.x / mag, to_center.y / mag))
        perp = Point2((-fwd.y, fwd.x))

        candidates: list[Point2] = [
            # Behind minerals (away from map center)
            Point2((base.x - fwd.x * 7, base.y - fwd.y * 7)),
            # Lateral left
            Point2((base.x + perp.x * 7, base.y + perp.y * 7)),
            # Lateral right
            Point2((base.x - perp.x * 7, base.y - perp.y * 7)),
            # Diagonals
            Point2((base.x - fwd.x * 5 + perp.x * 5, base.y - fwd.y * 5 + perp.y * 5)),
            Point2((base.x - fwd.x * 5 - perp.x * 5, base.y - fwd.y * 5 - perp.y * 5)),
            # Fallback: close behind base
            Point2((base.x - fwd.x * 4, base.y - fwd.y * 4)),
        ]

        placed_pos: Optional[Point2] = None
        for candidate in candidates:
            try:
                pos = await self.find_placement(UnitID.SPIRE, near=candidate, placement_step=2)
                if pos is not None:
                    placed_pos = pos
                    break
            except Exception:
                continue

        if placed_pos is None:
            return  # retry next step

        worker = self.mediator.select_worker(target_position=placed_pos)
        if not worker:
            return

        self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
        self._spire_worker_tag = worker.tag
        self.mediator.build_with_specific_worker(
            worker=worker,
            structure_type=UnitID.SPIRE,
            pos=placed_pos,
            building_purpose=BuildingPurpose.NORMAL_BUILDING,
        )
        self.spire_ordered = True
        await self.chat_send("Tag: Spire_Ordered")



    async def make_zerglings(self):
        if "Flying_Structures" not in self.enemy_strategy:
            if self.minerals >700:
                if self.vespene < 25:
                    self.train(UnitID.ZERGLING)


    async def find_liberator(self):
        if self.liberatorFound == False:
            for unit in self.enemy_units:
                if unit.name == 'Liberator':
                    self.liberatorFound = True
                    break
            if self.liberatorFound:
                await self.chat_send("Tag: Liberator")
                self.enemy_strategy.append("Liberator")


    async def turnOffSpawningControllerOnEarlyGame(self):
        if self.build_order_runner.build_completed == False:
            self.spawn_inhibitors.add("build_order_not_complete")
        else:
            self.spawn_inhibitors.discard("build_order_not_complete")

    async def turnOffSpeedMining(self):
        if self.speedMiningOn == True:
            self.speedMiningOn = False


    async def harass_worker_proxy_barracks(self):
        worker_scouts: Units = self.mediator.get_units_from_role(
            role=UnitRole.BUILD_RUNNER_SCOUT, unit_type=self.worker_type
        )
        
        for scout in worker_scouts:
            self.mediator.switch_roles(
                from_role=UnitRole.BUILD_RUNNER_SCOUT, to_role=UnitRole.HARASSING)

        worker_scouts: Units = self.mediator.get_units_from_role(
            role=UnitRole.HARASSING, unit_type=self.worker_type
        )


        # Adicionar todos os SCVs encontrados na lista de scout_targets
        for unit in self.enemy_units:
            if unit.name == 'SCV' and unit.tag not in self.scout_targets:
                self.scout_targets[unit.tag] = unit
    
        # Remover SCVs que não estão mais na lista de unidades inimigas
        self.scout_targets = {tag: target for tag, target in self.scout_targets.items() if target in self.enemy_units}
    
        for scout in worker_scouts:
            # Se a lista de scout_targets não estiver vazia, atacar o primeiro SCV da lista
            if self.scout_targets:
                first_target_tag = next(iter(self.scout_targets))
                first_target = self.scout_targets[first_target_tag]
                scout.attack(first_target)
            else:
                # Se a lista de scout_targets estiver vazia, atacar a primeira estrutura inimiga
                if self.enemy_structures:
                    scout.attack(self.enemy_structures.first.position)


    async def is_3_base_terran(self):
        if self.time > 300:
            if self.enemy_has_3_bases == False:
                if self.mediator.get_enemy_has_base_outside_natural == True:
                    await self.chat_send("Tag: 3_Base_Terran")
                    self.enemy_strategy.append("3_Base_Terran")
                    self.enemy_has_3_bases = True





    async def spread_overlords(self):
        expansion_locations = list(self.expansion_locations_list)
        overlord_tags = list(self.my_overlords.keys())  # Obter as tags dos Overlords em ordem
    
        # Iterar sobre cada expansão e atribuir um Overlord
        for i, expansion in enumerate(expansion_locations):
            if i < len(overlord_tags):
                overlord_tag = overlord_tags[i]
                overlord = self.my_overlords[overlord_tag]
    
                # Enviar o Overlord para a expansão apenas se ele não estiver se movendo
                #if not overlord.is_moving:
                self.do(overlord.move(expansion))

    async def is_worker_rush(self):
        if self.enemy_went_worker_rush == False:
            if self.mediator.get_enemy_worker_rushed == True:
                await self.chat_send("Tag: Worker_Rush")
                self.enemy_strategy.append("Worker_Rush")
                self.enemy_went_worker_rush = True


    async def change_to_bo_DefensiveVsProxyBarracks(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("DefensiveVsProxyBarracks")
            self.bo_changed = True


    async def force_complete_build_order(self):
        if self.build_order_runner.build_completed == False:
            if self.time > 300:
                self.build_order_runner.set_build_completed()
                await self.chat_send("Tag: Build_Completed")
                self.enemy_strategy.append("Force_Build_Completed")


    async def stop_collecting_gas(self):
        if not "2_Proxy_Gateway" in self.enemy_strategy:
            if self.stop_getting_gas == False:
                if self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED) or "Proxy_Stargate" in self.enemy_strategy:
                    print("Chamando set_workers_per_gas com amount=0")
                    self.mediator.set_workers_per_gas(amount=0)
                    self.workers_for_gas = 0
                    self.stop_getting_gas = True
                    #self.stop_getting_gas = True


    async def burrow_infestors(self):
        # Burrow the roaches when they are low health
        for infestor in self.units(UnitID.INFESTOR):
            if infestor.energy <= 75:
                infestor(AbilityId.BURROWDOWN_INFESTOR)


        for burrowed_infestor in self.units(UnitID.INFESTORBURROWED):
            if burrowed_infestor.energy > 75:
                burrowed_infestor(AbilityId.BURROWUP_INFESTOR)


    async def create_queens_after_build_order(self):
        if self.build_order_runner.build_completed:
            for th in self.townhalls.ready:
                # Check if the number of queens is less than the number of townhalls
                if len(self.units(UnitID.QUEEN)) <= len(self.townhalls.ready) + 1:
                    # Check if we're not already training a queen
                    if not self.already_pending(UnitID.QUEEN):
                        # If we're not, train a queen
                        self.do(th.train(UnitID.QUEEN))


    async def change_to_bo_TwelvePool(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("TwelvePool")
            self.bo_changed = True

    async def zergling_scout(self):
        if self.time < 146:
            for zergling in self.units(UnitID.ZERGLING):
                zergling.move(self.enemy_start_locations[0])

    async def make_overseer(self):
        if not self.structures(UnitID.LAIR):
            return

        banshee_count = sum(1 for u in self.enemy_units if u.name == 'Banshee')
        target_count = max(1, banshee_count)

        current_count = (
            self.units(UnitID.OVERSEER).amount
            + self.units(UnitID.OVERLORDCOCOON).amount
        )

        if current_count >= target_count:
            return

        if not self.can_afford(UnitID.OVERSEER):
            return

        # Morph any overlord that is not the first (scouting) overlord
        # and not the second overlord reserved for changeling scouting
        first_overlord_tag = getattr(self.first_overlord, 'tag', self.first_overlord)
        second_overlord_tag = self.tag_second_overlord
        for overlord in self.units(UnitID.OVERLORD):
            if overlord.tag != first_overlord_tag and overlord.tag != second_overlord_tag:
                overlord(AbilityId.MORPH_OVERSEER)
                break


 

    async def assign_overseer(self):
        overseers = self.units(UnitID.OVERSEER).ready
        if not overseers:
            return

        if self.EnemyRace == Race.Terran:
            banshees = [unit for unit in self.enemy_units if unit.name == 'Banshee']
            banshee_by_tag = {b.tag: b for b in banshees}

            # Atualiza posições conhecidas das banshees visíveis
            for b in banshees:
                self.last_known_banshee_positions[b.tag] = b.position
                self.last_known_banshee_frames[b.tag] = self.state.game_loop

            # Remove posições não vistas há mais de 500 frames (~22 segundos)
            stale_tags = [
                tag for tag, frame in self.last_known_banshee_frames.items()
                if self.state.game_loop - frame > 500
            ]
            for tag in stale_tags:
                self.last_known_banshee_positions.pop(tag, None)
                self.last_known_banshee_frames.pop(tag, None)

            # Remove atribuições de overseers que já não existem
            active_overseer_tags = {o.tag for o in overseers}
            self.overseer_banshee_assignments = {
                ot: bt for ot, bt in self.overseer_banshee_assignments.items()
                if ot in active_overseer_tags
            }

            # Remove atribuições para banshees sem posição conhecida (mortas/expiradas)
            self.overseer_banshee_assignments = {
                ot: bt for ot, bt in self.overseer_banshee_assignments.items()
                if bt in banshee_by_tag or bt in self.last_known_banshee_positions
            }

            # Banshee tags já atribuídas
            assigned_banshee_tags = set(self.overseer_banshee_assignments.values())

            # Atribui overseers sem banshee a banshees ainda não atribuídas
            # Exclui o overseer reservado para o scouting com changeling
            scout_overseer_tag = self.tag_second_overlord
            unassigned_overseers = [
                o for o in overseers
                if o.tag not in self.overseer_banshee_assignments and o.tag != scout_overseer_tag
            ]
            unassigned_banshees = [b for b in banshees if b.tag not in assigned_banshee_tags]

            for overseer in unassigned_overseers:
                if unassigned_banshees:
                    # Escolhe a banshee mais próxima deste overseer
                    target = min(unassigned_banshees, key=lambda b: overseer.distance_to(b))
                    self.overseer_banshee_assignments[overseer.tag] = target.tag
                    unassigned_banshees.remove(target)

            # Move cada overseer para sua banshee atribuída
            for overseer in overseers:
                banshee_tag = self.overseer_banshee_assignments.get(overseer.tag)
                if banshee_tag is None:
                    continue

                if banshee_tag in banshee_by_tag:
                    # Banshee visível: segue diretamente
                    self.do(overseer.move(banshee_by_tag[banshee_tag].position))
                elif banshee_tag in self.last_known_banshee_positions:
                    # Banshee invisível: vai para última posição conhecida
                    self.do(overseer.move(self.last_known_banshee_positions[banshee_tag]))
            return

        if self.EnemyRace == Race.Zerg:
            # 2. Caso contrário, siga a roach mais próxima da base inimiga
            roaches = self.units(UnitID.ROACH).ready
            if roaches:
                enemy_main = self.enemy_start_locations[0]
                target_roach = min(roaches, key=lambda r: r.distance_to(enemy_main))
                for overseer in overseers:
                    if overseer.distance_to(target_roach) > 2:
                        self.do(overseer.move(target_roach.position))


    async def build_one_spine_crawler(self):
        if self.rally_point_set == True:
            if self.structures(UnitID.SPINECRAWLER).amount == 0 and not self.already_pending(UnitID.SPINECRAWLER):
                if self.tag_worker_build_spine_crawler == 0:
                    if self.can_afford(UnitID.SPINECRAWLER):
                        my_base_location = self.mediator.get_own_nat
                        # Send the second Overlord in front of second base to scout
                        target = my_base_location.position.towards(self.game_info.map_center, 4)                   
                        #await self.build(UnitID.HYDRALISKDEN, near=target)
                        if worker := self.mediator.select_worker(target_position=target):                
                            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                            self.tag_worker_build_spine_crawler = worker
                            #self.mediator.build_with_specific_worker(worker, UnitID.HATCHERY, target, BuildingPurpose.NORMAL_BUILDING)
                            self.mediator.build_with_specific_worker(worker=self.tag_worker_build_spine_crawler, structure_type=UnitID.SPINECRAWLER, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)
                            print("first Spine Crawler")

    async def build_2_spine_crawlers(self):
        if not self.rally_point_set:
            return

        # Base de referência: segunda base se existir, senão a primeira
        base = getattr(self, "second_base", None) or self.first_base
        base_pos = base.position

        # Vetor da base em direção ao centro do mapa (frente)
        to_center = self.game_info.map_center - base_pos
        mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
        dir_unit = Point2((to_center.x / mag, to_center.y / mag))

        # Vetor perpendicular (eixo da linha dos spines)
        line_unit = Point2((-dir_unit.y, dir_unit.x))

        # Parâmetros da linha
        forward_offset = 6.0   # quão à frente da base
        spacing = 2.5          # distância lateral entre spines

        # Âncora (ponto central da linha)
        anchor = base_pos.towards(self.game_info.map_center, forward_offset)

        # Slots-alvo em linha: somente esquerda e direita (exclui o centro/âncora)
        # Empurra cada slot +1.0 na direção de construção (afastando da segunda base)
        slot_left   = Point2((anchor.x - line_unit.x * spacing, anchor.y - line_unit.y * spacing))
        slot_right  = Point2((anchor.x + line_unit.x * spacing, anchor.y + line_unit.y * spacing))
        slots = [slot_left, slot_right]

        def has_spine_near(p: Point2, radius: float = 2.0) -> bool:
            # Considera spines prontos e em construção
            return any(s.distance_to(p) <= radius for s in self.structures(UnitID.SPINECRAWLER))

        async def place_spine_at(pos: Point2) -> Point2 | None:
            # Se não houver creep exato, tenta pequenos ajustes laterais na própria linha
            candidate = pos
            if not self.has_creep(candidate):
                found = None
                for d in (0.5, 1.0, 1.5, 2.0):
                    for sign in (1, -1):
                        test = Point2((pos.x + line_unit.x * d * sign, pos.y + line_unit.y * d * sign))
                        if self.has_creep(test):
                            found = test
                            break
                    if found:
                        break
                if found:
                    candidate = found
                else:
                    # último recurso: recua levemente em direção à base para pegar creep
                    candidate = pos.towards(base_pos, 1.0)

            # Refina com find_placement para evitar colisão/minérios
            try:
                placed = await self.find_placement(UnitID.SPINECRAWLER, near=candidate, placement_step=1)
                # Se achou posição, garantir espaçamento mínimo de 2.0 contra spines existentes/em construção
                if placed is not None:
                    min_spacing = 2.0
                    def spaced_ok(p: Point2) -> bool:
                        return all(p.distance_to(s.position) >= min_spacing for s in self.structures(UnitID.SPINECRAWLER))

                    if spaced_ok(placed):
                        return placed

                    # Tenta deslocar lateralmente ao longo da linha para ganhar espaço
                    for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                        for sign in (1, -1):
                            shifted = Point2((placed.x + line_unit.x * d * sign, placed.y + line_unit.y * d * sign))
                            if not self.has_creep(shifted):
                                continue
                            try:
                                alt = await self.find_placement(UnitID.SPINECRAWLER, near=shifted, placement_step=1)
                            except Exception:
                                alt = None
                            if alt is not None and spaced_ok(alt):
                                return alt
            except Exception:
                pass
            # fallback: retorna candidato somente se houver creep e espaçamento adequado
            if self.has_creep(candidate):
                min_spacing = 2.0
                if all(candidate.distance_to(s.position) >= min_spacing for s in self.structures(UnitID.SPINECRAWLER)):
                    return candidate
            return None

        # Mapeia cada slot para o atributo de tag do seu worker (apenas 2)
        # Usamos os atributos de 2º e 3º para não conflitar com o centro
        slot_attr = {
            0: "tag_worker_build_2nd_spine_crawler",  # esquerda
            1: "tag_worker_build_3rd_spine_crawler",  # direita
        }

        # Constrói exatamente 2 spines, mantendo alinhamento em linha
        for idx, pos in enumerate(slots):
            # Ordem sequencial: só inicia o 2º após o 1º começar
            if idx == 1 and not (has_spine_near(slots[0]) or getattr(self, slot_attr[0]) != 0):
                break
            # Pula se já existe um spine nesse slot
            if has_spine_near(pos):
                continue
            # Pula se já temos worker associado a esse slot
            if getattr(self, slot_attr[idx]) != 0:
                continue
            # Verifica recursos
            if not self.can_afford(UnitID.SPINECRAWLER):
                break

            build_pos = await place_spine_at(pos)
            if not build_pos:
                continue

            if worker := self.mediator.select_worker(target_position=build_pos):
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                setattr(self, slot_attr[idx], worker)
                self.mediator.build_with_specific_worker(
                    worker=getattr(self, slot_attr[idx]),
                    structure_type=UnitID.SPINECRAWLER,
                    pos=build_pos,
                    building_purpose=BuildingPurpose.NORMAL_BUILDING,
                )
                # print opcional:
                # print(f"Spine Crawler slot {idx} at {build_pos}")

    async def make_changeling(self):
        # Filtra apenas overseers prontos e com energia suficiente
        for overseer in self.units(UnitID.OVERSEER).ready:
            if overseer.energy > 50 and overseer.is_ready:
                overseer(AbilityId.SPAWNCHANGELING_SPAWNCHANGELING)

    async def move_changeling(self):
        for changeling in self.units(UnitID.CHANGELING):
            if changeling.distance_to(self.enemy_start_locations[0]) > 20:
                changeling.move(self.enemy_start_locations[0])
            else:
                changeling.move(self.game_info.map_center)




    async def is_ling_rush(self):
        if self.enemy_went_ling_rush == False:
            if self.mediator.get_enemy_ling_rushed == True:
                await self.chat_send("Tag: Ling_Rush")
                self.enemy_strategy.append("Ling_Rush")
                self.enemy_went_ling_rush = True


    async def stop_build_order(self):
        if self.build_order_runner.build_completed == False:
            self.build_order_runner.set_build_completed()
            await self.chat_send("Tag: Build_Completed")
            self.enemy_strategy.append("Force_Build_Completed")



    async def is_twelve_pool(self):
        if "12_Pool" not in self.enemy_strategy:
        #verify if the protoss opponent has only one base. If so, it is an agressive terran and build a spine crawler
            if self.time < 82:
                found_pool = False
                for unit in self.enemy_structures:
                    if unit.name == 'SpawningPool':
                        if unit.build_progress == 1:
                            found_pool = True
                            break  # Breake the loop if find the Nexus
                if found_pool:
                    await self.chat_send("Tag: 12_Pool")
                    self.enemy_strategy.append("12_Pool")



    async def change_to_bo_VsOneBaseRandomProtoss(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("VsOneBaseRandomProtoss")
            self.bo_changed = True


    async def build_roach_warren_failed(self):
        if self.time > 190:
            if self.structures(UnitID.SPAWNINGPOOL).ready:
                if self.structures(UnitID.ROACHWARREN).amount == 0 and not self.already_pending(UnitID.ROACHWARREN):
                    if self.tag_worker_build_roach_warren == 0:
                        if self.can_afford(UnitID.ROACHWARREN):
                            map_center = self.game_info.map_center
                            position_towards_map_center = self.start_location.towards(map_center, distance=5)
                            target = await self.find_placement(UnitID.ROACHWARREN, near=position_towards_map_center, placement_step=1)
                            #await self.build(UnitID.HYDRALISKDEN, near=target)
                            if worker := self.mediator.select_worker(target_position=target):                
                                self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                                self.tag_worker_build_roach_warren = worker
                                #self.mediator.build_with_specific_worker(worker, UnitID.HATCHERY, target, BuildingPurpose.NORMAL_BUILDING)
                                self.mediator.build_with_specific_worker(worker=self.tag_worker_build_roach_warren, structure_type=UnitID.ROACHWARREN, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)


    async def is_mass_marauder(self):
        if "Mass_Marauder" not in self.enemy_strategy:
            marauder_count = sum(1 for unit in self.enemy_units if unit.name == 'Marauder')
            if marauder_count >= 3:
                await self.chat_send("Tag: Mass_Marauder")
                self.enemy_strategy.append("Mass_Marauder")



    async def is_mass_liberator(self):
        if "Mass_Liberator" not in self.enemy_strategy:
            liberator_count = sum(1 for unit in self.enemy_units if unit.name == 'Liberator')
            if liberator_count >= 2:
                await self.chat_send("Tag: Mass_Liberator")
                self.enemy_strategy.append("Mass_Liberator")


    async def make_ravagers(self):
        # Mesmos gates que você já tinha
        # Só morfar se não houver spawn inhibitors ativos
        if self.vespene > 250 and self.structures(UnitID.ROACHWARREN).ready and not self.spawn_inhibitors:
            roaches: Units = self.units(UnitID.ROACH).ready
            if roaches.amount < 9:
                return
            if "Flying_Structures" in self.enemy_strategy:
                return

            # Ponto de referência: segunda base, senão fallback
            if getattr(self, "second_base", None):
                target_pos = self.second_base.position
            elif self.townhalls.ready:
                target_pos = self.townhalls.ready.closest_to(self.start_location).position
            elif self.townhalls:
                target_pos = self.townhalls.closest_to(self.start_location).position
            else:
                target_pos = self.start_location  # último fallback seguro

            # Escolhe a roach mais próxima do target_pos
            roach = min(roaches, key=lambda r: r.distance_to(target_pos))

            # Checa se pode morfar agora (evita spam/cooldown)
            abilities = await self.get_available_abilities(roach)
            if AbilityId.MORPHTORAVAGER_RAVAGER in abilities:
                roach(AbilityId.MORPHTORAVAGER_RAVAGER)
                if "Ravager" not in self.enemy_strategy:
                    await self.chat_send("Tag: Ravager")
                    self.enemy_strategy.append("Ravager")

                    
    async def build_plus_one_roach_armor(self):
        # Se já pesquisou ZERGMISSILEWEAPONSLEVEL1, pesquisa ZERGGROUNDARMORSLEVEL1
        # O SpawnController só fica bloqueado até o upgrade COMEÇAR a ser pesquisado
        if self.structures(UnitID.EVOLUTIONCHAMBER).ready and self.structures(UnitID.SPAWNINGPOOL).ready:
            if UpgradeId.ZERGMISSILEWEAPONSLEVEL1 in self.state.upgrades:
                if UpgradeId.ZERGGROUNDARMORSLEVEL1 not in self.state.upgrades:
                    if self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                        # Já está sendo pesquisado: libera o SpawnController
                        self.spawn_inhibitors.discard("researching_armor")
                    elif self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL2):
                        # Já está sendo pesquisado: libera o SpawnController
                        self.spawn_inhibitors.discard("researching_armor")    
                    else:
                        # Ainda não começou: bloqueia e inicia quando puder
                        self.spawn_inhibitors.add("researching_armor")
                        if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                            self.research(UpgradeId.ZERGGROUNDARMORSLEVEL1)
                else:
                    self.spawn_inhibitors.discard("researching_armor")
            else:
                self.spawn_inhibitors.discard("researching_armor")




    async def is_mass_widow_mine(self):
        """
        Registra cada Widow Mine (burrowed ou não) apenas uma vez pelo tag.
        A mesma unidade ao alternar entre WIDOWMINE <-> WIDOWMINEBURROWED mantém o mesmo tag.
        """
        if "Mass_Widow_Mine" in self.enemy_strategy:
            return

        # Itera minas vistas neste frame
        for enemy in self.enemy_units.of_type({UnitID.WIDOWMINE, UnitID.WIDOWMINEBURROWED}):
            if enemy.tag not in self.enemy_widow_mines:
                # registra primeira vez que vimos essa mina
                self.enemy_widow_mines[enemy.tag] = enemy.type_id

        if len(self.enemy_widow_mines) >= 3:
            await self.chat_send("Tag: Mass_Widow_Mine")
            self.enemy_strategy.append("Mass_Widow_Mine")


    async def change_to_bo_Terran_Agressive(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("TerranAgressive")
            self.bo_changed = True
            self._begin_attack_at_supply = 90


    async def is_mid_game(self):
        if self.time > 440:
            if self.mid_game == False:
                await self.chat_send("Tag: Mid_Game")
                self.enemy_strategy.append("Mid_Game")
                self.mid_game = True
                self._begin_attack_at_supply = 50


    async def mid_game_protocol(self):
        if not self.mid_game:
            return

        if "Late_Game" in self.enemy_strategy:
            return
        drone_count = self.units(UnitID.DRONE).amount
        bases_started = self.townhalls.ready.amount + math.ceil(self.already_pending(UnitID.HATCHERY))

        # Do not keep army production locked forever if expansions are delayed/denied.
        if drone_count >= 57 or bases_started >= 4:
            self.spawn_inhibitors.discard("mid_game_expanding")
            self.mid_game_expansion_done = True
        elif not self.mid_game_expansion_done:
            self.spawn_inhibitors.add("mid_game_expanding")

        # Vs Terran: once we reach the worker target, force Lair tech first.
        # build_lair() adds the "building_lair" inhibitor while saving resources,
        # which pauses worker production through BuildWorkersNoExpand.
        if self.EnemyRace == Race.Terran and drone_count >= 57:
            await self.build_lair()

        macro_plan: MacroPlan = MacroPlan()

        macro_plan.add(ExpansionController(to_count=4, max_pending=2))
        macro_plan.add(BuildWorkersNoExpand(to_count=57))
        macro_plan.add(GasBuildingController(to_count=7, max_pending=2))
        self.register_behavior(macro_plan)

    async def make_roach_speed(self):
        if UpgradeId.TUNNELINGCLAWS in self.state.upgrades:
            if UpgradeId.GLIALRECONSTITUTION not in self.state.upgrades:
                if not self.already_pending_upgrade(UpgradeId.GLIALRECONSTITUTION):
                    self.spawn_inhibitors.add("researching_roach_speed")
                    self.research(UpgradeId.GLIALRECONSTITUTION)
                    return
        self.spawn_inhibitors.discard("researching_roach_speed")


    async def is_mass_banshee(self):
        if "Mass_Banshee" not in self.enemy_strategy:
            for unit in self.enemy_units:
                if unit.name == 'Banshee':
                    if unit.tag not in self.enemy_banshees:
                        self.enemy_banshees[unit.tag] = unit.type_id
            if len(self.enemy_banshees) >= 3:
                await self.chat_send("Tag: Mass_Banshee")
                self.enemy_strategy.append("Mass_Banshee")



    async def change_to_bo_Bunker_Rush(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("BunkerRush")
            self.bo_changed = True


    async def change_to_bo_Protoss_Agressive(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("Protoss_Agressive")
            self.bo_changed = True


    async def build_infestation_pit(self):
        if not self.structures(UnitID.LAIR).ready:
            return

        if self.infestation_pit_ordered:
            return

        if self.structures(UnitID.INFESTATIONPIT) or self.already_pending(UnitID.INFESTATIONPIT):
            self.infestation_pit_ordered = True
            return

        self.macro_plan.add(
            BuildStructure(
                base_location=self.first_base.position,
                structure_id=UnitID.INFESTATIONPIT,
                to_count=1,
            )
        )
        self.register_behavior(self.macro_plan)
        self.infestation_pit_ordered = True
        await self.chat_send("Tag: Infestation_Pit_Ordered")




    async def make_macro_hatch(self):
        # Só constrói se tiver mais de 275 de minério e o worker ainda não foi alocado
        if self.minerals < 275:
            return

        if self.taf_worker_build_macro_hatch != 0:
            return

        if not self.can_afford(UnitID.HATCHERY):
            return

        # O rally point fica ~6 tiles em direção ao centro, então candidatos em
        # outras direções: atrás dos minerais e lateral à base.
        candidates: list[Point2] = []

        # 1) Atrás da linha de minerais (direção oposta ao mapa)
        mineral_positions = self.mediator.get_behind_mineral_positions(
            th_pos=self.first_base.position
        )
        if mineral_positions:
            candidates.append(mineral_positions[0])

        # 2) Lateral esquerda / direita em relação à direção do mapa
        to_center = self.game_info.map_center - self.first_base.position
        mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
        perp = Point2((-to_center.y / mag, to_center.x / mag))
        for sign in (1, -1):
            candidates.append(Point2((
                self.first_base.position.x + perp.x * sign * 8,
                self.first_base.position.y + perp.y * sign * 8,
            )))

        # 3) Fallback: mais perto da base em direção ao mapa (evita o rally point exato)
        candidates.append(self.first_base.position.towards(self.game_info.map_center, 10))

        placed_pos: Optional[Point2] = None
        for candidate in candidates:
            try:
                placed = await self.find_placement(UnitID.HATCHERY, near=candidate, placement_step=3)
                if placed is not None:
                    placed_pos = placed
                    break
            except Exception:
                continue

        if placed_pos is None:
            return  # tenta de novo no próximo step

        if worker := self.mediator.select_worker(target_position=placed_pos):
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
            self.taf_worker_build_macro_hatch = worker
            self.macro_hatch_pos = placed_pos
            self.mediator.build_with_specific_worker(
                worker=self.taf_worker_build_macro_hatch,
                structure_type=UnitID.HATCHERY,
                pos=placed_pos,
                building_purpose=BuildingPurpose.NORMAL_BUILDING,
            )
        

    async def is_bc(self):
        """
        Make air defense if enemy is making a lot of banshees
        """
        if "Battlecruiser" in self.enemy_strategy:
            return

        # Detecta Fusion Core — inimigo está fazendo Battlecruisers
        if self.enemy_structures.of_type({UnitID.FUSIONCORE}):
            await self.chat_send("Tag: Battlecruiser")
            self.enemy_strategy.append("Battlecruiser")
            self.build_order_runner.set_build_completed()
            return

        # Itera battlecruisers vistas neste frame
        for enemy in self.enemy_units.of_type({UnitID.BATTLECRUISER}):
            if enemy.tag not in self.enemy_battlecruisers:
                self.enemy_battlecruisers[enemy.tag] = enemy.type_id

        if len(self.enemy_battlecruisers) >= 1:
            await self.chat_send("Tag: Battlecruiser")
            self.enemy_strategy.append("Battlecruiser")


    async def is_mass_tank(self):
        if "Mass_Tank" not in self.enemy_strategy:
            tank_count = sum(1 for unit in self.enemy_units if unit.name == 'SiegeTankSieged' or unit.name == 'SiegeTank')
            if tank_count >= 3:
                await self.chat_send("Tag: Mass_Tank")
                self.enemy_strategy.append("Mass_Tank")

    async def check_nydus_spot(self):
        nydus_spot = self.mediator.get_primary_nydus_own_main
        if nydus_spot is None:
            return
        if isinstance(nydus_spot, (list, tuple, np.ndarray)) and len(nydus_spot) == 0:
            return

        try:
            if hasattr(nydus_spot, "x") and hasattr(nydus_spot, "y"):
                target = Point2((float(nydus_spot.x), float(nydus_spot.y)))
            else:
                x, y = nydus_spot
                target = Point2((float(x), float(y)))
        except Exception:
            return

        overlords: Units = self.units(UnitID.OVERLORD)
        if not overlords:
            return

        third_overlord: Optional[Unit] = None
        if self.tag_third_overlord:
            third_overlord = overlords.find_by_tag(self.tag_third_overlord)
            if not third_overlord:
                self.tag_third_overlord = 0
                self.nydus_spot_set = False

        if not third_overlord:
            alive_tags = set(overlords.tags)
            ordered_tags = [tag for tag in self.my_overlords.keys() if tag in alive_tags]
            if len(ordered_tags) >= 3:
                third_tag = ordered_tags[2]
                third_overlord = overlords.find_by_tag(third_tag)
                if third_overlord:
                    self.tag_third_overlord = third_tag
            elif overlords.amount >= 3:
                third_overlord = sorted(overlords, key=lambda o: o.tag)[2]
                self.tag_third_overlord = third_overlord.tag

        if not third_overlord:
            return

        if not self.nydus_spot_set or third_overlord.distance_to(target) > 2:
            self.do(third_overlord.move(target))
            self.nydus_spot_set = True

    async def build_safe_spine(self):
        if not getattr(self, "second_base", None):
            return

        # Evita construir se já houver spine na segunda base ou se já tiver worker alocado
        if self.structures(UnitID.SPINECRAWLER).amount > 0 or self.already_pending(UnitID.SPINECRAWLER):
            return
        if getattr(self, "tag_worker_build_spine_crawler", 0) != 0:
            return
        if not self.can_afford(UnitID.SPINECRAWLER):
            return

        base = self.second_base
        positions = self.mediator.get_behind_mineral_positions(th_pos=base.position)
        if not positions:
            return

        # "positions" costuma ficar atrás dos minerais; queremos entre base e minerais
        mineral_ref = positions[0]
        # ponto entre a base e a linha de minério (ligeiramente mais perto da base)
        target = base.position.towards(mineral_ref, 3)

        # garante creep: se não tiver, aproxima mais da base
        if not self.has_creep(target):
            target = base.position.towards(mineral_ref, 2)

        # refina com find_placement
        try:
            placed = await self.find_placement(UnitID.SPINECRAWLER, near=target, placement_step=1)
        except Exception:
            placed = None

        build_pos = placed if placed else target
        if not self.has_creep(build_pos):
            return

        if worker := self.mediator.select_worker(target_position=build_pos):
            self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
            self.tag_worker_build_spine_crawler = worker
            self.mediator.build_with_specific_worker(
                worker=self.tag_worker_build_spine_crawler,
                structure_type=UnitID.SPINECRAWLER,
                pos=build_pos,
                building_purpose=BuildingPurpose.NORMAL_BUILDING,
            )


    async def change_to_bo_Vs_Ling_Rush(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("VsLingRush")
            self.bo_changed = True



    async def is_mid_game_vs_protoss(self):
        if "2_Base_Protoss" in self.enemy_strategy:
             if self.time > 360:
                if self.mid_game == False:
                    await self.chat_send("Tag: Mid_Game")
                    self.enemy_strategy.append("Mid_Game")
                    self.mid_game = True

        if "Protoss_Agressive" in self.enemy_strategy:
            if self.time > 480:
                if self.mid_game == False:
                    await self.chat_send("Tag: Mid_Game")
                    self.enemy_strategy.append("Mid_Game")
                    self.mid_game = True



    async def mid_game_vs_protoss_protocol(self):
        if self.mid_game:
            if "Protoss_Agressive" in self.enemy_strategy:
                #bases = self.townhalls.ready
                #if self.workers.amount < 54:
                    #if not self.already_pending(UnitID.HATCHERY):
                        #self.SapwnControllerOn = False
                self.register_behavior(ExpansionController(to_count=4, max_pending=2))
                self.register_behavior(BuildWorkersNoExpand(to_count=55))           
                self.register_behavior(GasBuildingController(to_count=5, max_pending=2))

            if "2_Base_Protoss" in self.enemy_strategy:

                if self.workers.amount < 55:
                    self.spawn_inhibitors.add("mid_game_vs_protoss_workers")
                    self.register_behavior(ExpansionController(to_count=5, max_pending=3))
                    self.register_behavior(BuildWorkersNoExpand(to_count=55))           
                else:
                    self.spawn_inhibitors.discard("mid_game_vs_protoss_workers")

            if "Cannon_Rush" in self.enemy_strategy:

                if self.workers.amount < 55:
                    self.spawn_inhibitors.add("mid_game_vs_protoss_workers")
                    self.register_behavior(ExpansionController(to_count=5, max_pending=3))
                    self.register_behavior(BuildWorkersNoExpand(to_count=55))           

                      
            
            



    async def build_missle_upgrades(self):
        chambers = self.structures(UnitID.EVOLUTIONCHAMBER).ready
        if not (chambers and self.structures(UnitID.SPAWNINGPOOL).ready):
            return

        chambers_count = chambers.amount

        # --- Level 1: missile + armor in parallel (need 2 chambers for both) ---
        missile1_done = UpgradeId.ZERGMISSILEWEAPONSLEVEL1 in self.state.upgrades
        armor1_done   = UpgradeId.ZERGGROUNDARMORSLEVEL1 in self.state.upgrades

        if not missile1_done:
            if self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL1):
                self.spawn_inhibitors.discard("researching_missle_level_1")
            else:
                self.spawn_inhibitors.add("researching_missle_level_1")
                if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL1):
                    self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL1)
        else:
            self.spawn_inhibitors.discard("researching_missle_level_1")

        if not armor1_done:
            if self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                self.spawn_inhibitors.discard("researching_armor_level_1")
            elif missile1_done or chambers_count >= 2:
                # Chamber is free (missile done) or we have a spare chamber — order armor
                self.spawn_inhibitors.add("researching_armor_level_1")
                if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL1):
                    self.research(UpgradeId.ZERGGROUNDARMORSLEVEL1)
            else:
                # Single chamber still busy with missile: wait, don't block spawn yet
                self.spawn_inhibitors.discard("researching_armor_level_1")
        else:
            self.spawn_inhibitors.discard("researching_armor_level_1")

        if not missile1_done or not armor1_done:
            return

        # --- Level 2 upgrades require Lair tech; Hive also satisfies this ---
        if not self.townhalls.of_type({UnitID.LAIR, UnitID.HIVE}).ready:
            return

        # --- Level 2: missile + armor in parallel (need 2 chambers for both) ---
        missile2_done = UpgradeId.ZERGMISSILEWEAPONSLEVEL2 in self.state.upgrades
        armor2_done   = UpgradeId.ZERGGROUNDARMORSLEVEL2 in self.state.upgrades

        if not missile2_done:
            if self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL2):
                self.spawn_inhibitors.discard("researching_missle_level_2")
            else:
                self.spawn_inhibitors.add("researching_missle_level_2")
                if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL2):
                    self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL2)
        else:
            self.spawn_inhibitors.discard("researching_missle_level_2")

        if not armor2_done:
            if self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL2):
                self.spawn_inhibitors.discard("researching_armor_level_2")
            elif missile2_done or chambers_count >= 2:
                # Chamber is free (missile done) or we have a spare chamber — order armor
                self.spawn_inhibitors.add("researching_armor_level_2")
                if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL2):
                    self.research(UpgradeId.ZERGGROUNDARMORSLEVEL2)
            else:
                # Single chamber still busy with missile: wait
                self.spawn_inhibitors.discard("researching_armor_level_2")
        else:
            self.spawn_inhibitors.discard("researching_armor_level_2")

        if not missile2_done or not armor2_done:
            return

        # --- Level 3 upgrades require Hive ---
        if not self.townhalls.of_type({UnitID.HIVE}).ready:
            self.spawn_inhibitors.discard("researching_missle_level_3")
            self.spawn_inhibitors.discard("researching_armor_level_3")
            return

        # --- Level 3: missile + armor in parallel (need 2 chambers for both) ---
        missile3_done = UpgradeId.ZERGMISSILEWEAPONSLEVEL3 in self.state.upgrades
        armor3_done   = UpgradeId.ZERGGROUNDARMORSLEVEL3 in self.state.upgrades

        if not missile3_done:
            if self.already_pending_upgrade(UpgradeId.ZERGMISSILEWEAPONSLEVEL3):
                self.spawn_inhibitors.discard("researching_missle_level_3")
            else:
                self.spawn_inhibitors.add("researching_missle_level_3")
                if self.can_afford(UpgradeId.ZERGMISSILEWEAPONSLEVEL3):
                    self.research(UpgradeId.ZERGMISSILEWEAPONSLEVEL3)
        else:
            self.spawn_inhibitors.discard("researching_missle_level_3")

        if not armor3_done:
            if self.already_pending_upgrade(UpgradeId.ZERGGROUNDARMORSLEVEL3):
                self.spawn_inhibitors.discard("researching_armor_level_3")
            elif missile3_done or chambers_count >= 2:
                # Chamber is free (missile done) or we have a spare chamber — order armor
                self.spawn_inhibitors.add("researching_armor_level_3")
                if self.can_afford(UpgradeId.ZERGGROUNDARMORSLEVEL3):
                    self.research(UpgradeId.ZERGGROUNDARMORSLEVEL3)
            else:
                # Single chamber still busy with missile: wait
                self.spawn_inhibitors.discard("researching_armor_level_3")
        else:
            self.spawn_inhibitors.discard("researching_armor_level_3")


    async def build_evolution_chamber(self):
        if not self.structures(UnitID.SPAWNINGPOOL).ready:
            return

        if self.time < 172:
            return 
        
        # Already exists or fully built — clear state and exit
        if self.structures(UnitID.EVOLUTIONCHAMBER):
            self.evolution_chamber_ordered = False
            self._evo_worker_tag = 0
            return

        # Already pending (under construction via ares or direct build)
        if self.already_pending(UnitID.EVOLUTIONCHAMBER):
            return

        # Track the dispatched worker; reset if it died so we can retry
        if self._evo_worker_tag:
            worker = self.workers.find_by_tag(self._evo_worker_tag)
            if worker:
                return  # worker is alive and on the way
            # Worker died or was reassigned — reset and try again
            self._evo_worker_tag = 0
            self.evolution_chamber_ordered = False

        if not self.can_afford(UnitID.EVOLUTIONCHAMBER):
            return

        # Build candidate positions around the main base
        base = self.first_base.position
        center = self.game_info.map_center
        to_center = center - base
        mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
        fwd = Point2((to_center.x / mag, to_center.y / mag))
        perp = Point2((-fwd.y, fwd.x))

        candidates: list[Point2] = [
            Point2((base.x - fwd.x * 6, base.y - fwd.y * 6)),           # behind base
            Point2((base.x + perp.x * 6, base.y + perp.y * 6)),          # lateral left
            Point2((base.x - perp.x * 6, base.y - perp.y * 6)),          # lateral right
            Point2((base.x - fwd.x * 4 + perp.x * 4, base.y - fwd.y * 4 + perp.y * 4)),
            Point2((base.x - fwd.x * 4 - perp.x * 4, base.y - fwd.y * 4 - perp.y * 4)),
            Point2((base.x - fwd.x * 3, base.y - fwd.y * 3)),            # close fallback
        ]

        placed_pos: Optional[Point2] = None
        for candidate in candidates:
            try:
                pos = await self.find_placement(UnitID.EVOLUTIONCHAMBER, near=candidate, placement_step=2)
                if pos is not None:
                    placed_pos = pos
                    break
            except Exception:
                continue

        if placed_pos is None:
            return  # retry next step

        worker = self.mediator.select_worker(target_position=placed_pos)
        if not worker:
            return

        self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
        self._evo_worker_tag = worker.tag
        self.mediator.build_with_specific_worker(
            worker=worker,
            structure_type=UnitID.EVOLUTIONCHAMBER,
            pos=placed_pos,
            building_purpose=BuildingPurpose.NORMAL_BUILDING,
        )
        self.evolution_chamber_ordered = True
        await self.chat_send("Tag: Evo_Chamber_Ordered")


    async def build_spores_vs_bc(self):
        """3 Spore Crawlers on the main base, 2 on each other base.

        Maximum priority: blocks unit spawn and force-completes the build order
        while any required spore is missing (not built yet or destroyed).
        Automatically dispatches workers to rebuild destroyed spores.
        """
        MAIN_TARGET = 3    # spores required at first base
        OTHER_TARGET = 2   # spores required at each other base
        NEAR_RADIUS = 7.0  # tiles radius to associate a spore with a base
        DISPATCH_COOLDOWN = 8.0  # seconds between dispatches to the same base

        all_spores = self.structures(UnitID.SPORECRAWLER)  # includes under construction

        def spores_at(base_pos: Point2) -> int:
            return sum(1 for s in all_spores if s.distance_to(base_pos) < NEAR_RADIUS)

        first_base = self.first_base
        other_bases = [th for th in self.townhalls.ready if th.tag != first_base.tag]

        first_need = max(0, MAIN_TARGET - spores_at(first_base.position))
        other_needs = [(th, max(0, OTHER_TARGET - spores_at(th.position))) for th in other_bases]
        total_needed = first_need + sum(n for _, n in other_needs)

        # Train 2 extra drones for each base whose spore quota is already met (once only)
        def train_extra_drones_for(base_tag: int) -> None:
            if base_tag in self._spore_bc_extra_drones_done:
                return
            trained = 0
            for larva in self.units(UnitID.LARVA):
                if trained >= 2:
                    break
                if self.can_afford(UnitID.DRONE):
                    larva.train(UnitID.DRONE)
                    trained += 1
            if trained >= 2:
                self._spore_bc_extra_drones_done.add(base_tag)

        if first_need == 0:
            train_extra_drones_for(first_base.tag)
        for th, needed in other_needs:
            if needed == 0:
                train_extra_drones_for(th.tag)

        if total_needed == 0:
            self.spawn_inhibitors.discard("building_spores_vs_bc")
            return

        # Maximum priority: block unit spawn and build order
        self.spawn_inhibitors.add("building_spores_vs_bc")
        if not self.build_order_runner.build_completed:
            self.build_order_runner.set_build_completed()

        async def place_spore(base: Unit, slot: int) -> Optional[Point2]:
            """Return a valid creep placement for spore slot `slot` near `base`."""
            positions = self.mediator.get_behind_mineral_positions(th_pos=base.position)
            mineral_ref = positions[0] if positions else None

            if mineral_ref:
                to_min = mineral_ref - base.position
                mag = (to_min.x ** 2 + to_min.y ** 2) ** 0.5 or 1.0
                fwd = Point2((to_min.x / mag, to_min.y / mag))
                perp = Point2((-fwd.y, fwd.x))
                # (forward offset, perpendicular offset) per slot
                # fwd points base → minerals; negative fwd = towards map center
                # Main base: slots 0,1 between minerals and base; slot 2 towards center
                # Other bases: slots 0,1 (one each side of mineral line)
                offsets = [
                    (3.0,  2.0),   # slot 0: mineral side, perpendicular left
                    (3.0, -2.0),   # slot 1: mineral side, perpendicular right
                    (-5.0, 0.0),   # slot 2: opposite side of base, towards map center
                    (3.0,  0.0),   # slot 3: mineral side, centered (fallback)
                    (-3.5, 2.5),   # slot 4: center side + perp (fallback)
                ]
                fx, py = offsets[slot % len(offsets)]
                center = Point2((
                    base.position.x + fwd.x * fx + perp.x * py,
                    base.position.y + fwd.y * fx + perp.y * py,
                ))
            else:
                center = base.position.towards(self.game_info.map_center, -3 - slot)

            if not self.has_creep(center):
                found = None
                for delta in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                    for adj in [
                        Point2((center.x + delta, center.y)),
                        Point2((center.x, center.y + delta)),
                    ]:
                        if self.has_creep(adj):
                            found = adj
                            break
                    if found:
                        break
                if found is None:
                    return None
                center = found

            try:
                placed = await self.find_placement(UnitID.SPORECRAWLER, near=center, placement_step=1)
            except Exception:
                placed = None

            result = placed if placed else center
            return result if self.has_creep(result) else None

        async def dispatch_spores(base: Unit, needed: int, existing: int) -> None:
            """Send workers to build `needed` spores at `base`."""
            # Throttle: avoid spamming workers to the same base every frame
            last = self._spore_bc_last_dispatch.get(base.tag, 0.0)
            if self.time - last < DISPATCH_COOLDOWN:
                return
            sent = 0
            for i in range(needed):
                if not self.can_afford(UnitID.SPORECRAWLER):
                    break
                build_pos = await place_spore(base, existing + i)
                if build_pos is None:
                    continue
                if worker := self.mediator.select_worker(target_position=build_pos):
                    self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                    self.mediator.build_with_specific_worker(
                        worker=worker,
                        structure_type=UnitID.SPORECRAWLER,
                        pos=build_pos,
                        building_purpose=BuildingPurpose.NORMAL_BUILDING,
                    )
                    sent += 1
            if sent:
                self._spore_bc_last_dispatch[base.tag] = self.time

        await dispatch_spores(first_base, first_need, spores_at(first_base.position))
        for th, needed in other_needs:
            await dispatch_spores(th, needed, spores_at(th.position))


    async def build_more_queens(self):
        if self.minerals < 1000:
            return
        if not self.can_afford(UnitID.QUEEN):
            return
        for th in self.townhalls.ready:
            if th.is_idle:
                th.train(UnitID.QUEEN)

    async def make_spines_vs_ling_rush(self):
        second_base_ready = (
            self.second_base is not None
            and self.second_base.is_ready
        )
        if second_base_ready:
            await self.build_spine_crawlers()
        else:
            await self.make_spines_on_main()



    async def is_cannon_rush(self):
        if "Cannon_Rush" in self.enemy_strategy:
            return

        expansion = self.mediator.get_own_nat

        for structure in self.enemy_structures:
            if (
                structure.type_id == UnitID.PHOTONCANNON
                and self.time < 100
                and structure.distance_to(expansion) < 20
            ):
                await self.chat_send("Tag: Cannon_Rush")
                self.enemy_strategy.append("Cannon_Rush")
                break
            if structure.type_id == UnitID.FORGE and self.time < 75:
                await self.chat_send("Tag: Cannon_Rush")
                self.enemy_strategy.append("Cannon_Rush")
                break

    async def change_to_bo_CannonRush(self):
        if self.bo_changed == False:
            self.build_order_runner.switch_opening("CannonRush")
            self.bo_changed = True

    async def emergency_supply_block(self):
        """Treina overlords diretamente via larva quando o supply está crítico.
        Usado como fallback quando o build order runner está travado esperando
        recursos e o AutoSupply não consegue agir."""
        if self.supply_left > 2 or self.supply_cap >= 200:
            return
        if not self.can_afford(UnitID.OVERLORD):
            return
        for larva in self.units(UnitID.LARVA):
            larva.train(UnitID.OVERLORD)
            break

    async def scout_enemy_base_with_changeling(self):
        second_ol_tag = self.tag_second_overlord
        if not second_ol_tag:
            return

        # Posição onde o overseer vai ficar para spawnar o changeling (nydus spot da base inimiga)
        try:
            nydus_spot = self.mediator.get_primary_nydus_enemy_main
            if hasattr(nydus_spot, "x") and hasattr(nydus_spot, "y"):
                spawn_pos: Point2 = Point2((float(nydus_spot.x), float(nydus_spot.y)))
            else:
                spawn_pos = Point2((float(nydus_spot[0]), float(nydus_spot[1])))
        except Exception:
            spawn_pos = self.enemy_start_locations[0]

        # Frente da base inimiga: destino do changeling
        enemy_base: Point2 = self.enemy_start_locations[0]
        enemy_front: Point2 = enemy_base.towards(self.game_info.map_center, 8)

        # --- Etapa 2: morfar o segundo overlord em overseer quando o Lair estiver pronto ---
        if not self.structures(UnitID.LAIR).ready:
            return

        # O overlord ainda não morfou
        overlord: Optional[Unit] = self.units(UnitID.OVERLORD).find_by_tag(second_ol_tag)
        if overlord and not self.scout_changeling_spawned:
            if self.can_afford(UnitID.OVERSEER):
                overlord(AbilityId.MORPH_OVERSEER)
            return

        # --- Etapa 3: spawnar changeling sempre que o overseer tiver energia suficiente ---
        overseer: Optional[Unit] = self.units(UnitID.OVERSEER).find_by_tag(second_ol_tag)
        if overseer:
            at_spawn = overseer.distance_to(spawn_pos) <= 5
            has_energy = overseer.is_ready and overseer.energy >= 50
            approaching_energy = overseer.is_ready and overseer.energy >= 45

            if has_energy:
                if at_spawn:
                    # Na posição correta e com energia: spawna o changeling
                    overseer(AbilityId.SPAWNCHANGELING_SPAWNCHANGELING)
                    self.scout_changeling_spawned = True
                else:
                    # Tem energia mas foi empurrado para fora (ex: marines): volta à posição
                    overseer.move(spawn_pos)
            elif approaching_energy and not at_spawn:
                # Energia quase suficiente (45+): volta à posição para estar pronto quando chegar a 50
                overseer.move(spawn_pos)

        # --- Etapa 4: mover todos os changelings para a frente da base inimiga ---
        CHANGELING_TYPES = {
            UnitID.CHANGELING,
            UnitID.CHANGELINGMARINE,
            UnitID.CHANGELINGMARINESHIELD,
            UnitID.CHANGELINGZERGLING,
            UnitID.CHANGELINGZERGLINGWINGS,
            UnitID.CHANGELINGZEALOT,
        }
        for changeling in self.units.of_type(CHANGELING_TYPES):
            if changeling.distance_to(enemy_front) > 3:
                changeling.move(enemy_front)


    async def defend_worker_rush(self):
        """Pull drones to fight back against an enemy worker rush."""
        if "Worker_Rush" not in self.enemy_strategy:
            return

        # Scan a 20-tile radius around our main for enemy workers
        defense_point = self.first_base.position
        enemy_near: Units = self.mediator.get_units_in_range(
            start_points=[defense_point],
            distances=20,
            query_tree=UnitTreeQueryType.AllEnemy,
        )[0]
        enemy_workers: Units = enemy_near.filter(lambda u: u.type_id in WORKER_TYPES)

        defending: Units = self.mediator.get_units_from_role(
            role=UnitRole.DEFENDING,
            unit_type=self.worker_type,
        )

        # No enemy workers visible → release all defenders back to mining
        if not enemy_workers:
            for drone in defending:
                self.mediator.assign_role(tag=drone.tag, role=UnitRole.GATHERING)
            return

        # Pull 2× as many drones as enemy workers (between 4 and 20)
        workers_needed: int = min(20, max(4, int(len(enemy_workers) * 2)))

        if len(defending) < workers_needed:
            defending_tags = {d.tag for d in defending}
            available: Units = self.workers.filter(lambda w: w.tag not in defending_tags)
            to_pull = workers_needed - len(defending)
            for drone in available[:to_pull]:
                self.mediator.assign_role(tag=drone.tag, role=UnitRole.DEFENDING)

        # Refresh list after potential new assignments
        defending = self.mediator.get_units_from_role(
            role=UnitRole.DEFENDING,
            unit_type=self.worker_type,
        )

        # Command each defending drone to attack the closest enemy worker
        for drone in defending:
            if drone.is_carrying_resource and self.townhalls:
                drone.return_resource()
                continue
            target: Unit = cy_closest_to(drone.position, enemy_workers)
            maneuver = CombatManeuver()
            maneuver.add(ShootTargetInRange(unit=drone, targets=enemy_workers))
            maneuver.add(AttackTarget(unit=drone, target=target))
            self.register_behavior(maneuver)



    async def worker_attack_cannon_rush(self):
        """Worker micro to fight back against a cannon rush.

        Selects 5 workers per photon cannon still under construction within 15
        tiles of the natural. As soon as at least 1 photon cannon is completed,
        ALL workers stop attacking and return to mining.
        """
        if "Cannon_Rush" not in self.enemy_strategy:
            return

        # Reference point: natural expansion position (where the second base would be),
        # regardless of whether it has actually been built yet.
        nat = self.mediator.get_own_nat
        base_pos: Point2 = Point2((float(nat.x), float(nat.y))) if hasattr(nat, "x") else Point2((float(nat[0]), float(nat[1])))

        # Workers currently assigned to attack cannons (stored by target cannon tag)
        if not hasattr(self, "_cannon_rush_assignments"):
            self._cannon_rush_assignments: dict[int, list[int]] = {}  # cannon_tag -> [worker_tags]

        # If any photon cannon near the natural is completed, release ALL workers immediately
        any_cannon_complete = any(
            s.is_ready and s.distance_to(base_pos) <= 15
            for s in self.enemy_structures.of_type(UnitID.PHOTONCANNON)
        )
        if any_cannon_complete:
            for worker_tags in self._cannon_rush_assignments.values():
                for worker_tag in worker_tags:
                    worker = self.workers.find_by_tag(worker_tag)
                    if worker:
                        self.mediator.assign_role(tag=worker_tag, role=UnitRole.GATHERING)
            self._cannon_rush_assignments.clear()
            return

        # Find photon cannons still under construction within 15 tiles of the natural
        cannons_under_construction = [
            s for s in self.enemy_structures.of_type(UnitID.PHOTONCANNON)
            if not s.is_ready and s.distance_to(base_pos) <= 15
        ]

        # Release workers whose target cannon is now gone (destroyed)
        active_cannon_tags = {c.tag for c in cannons_under_construction}
        stale_cannon_tags = [tag for tag in self._cannon_rush_assignments if tag not in active_cannon_tags]
        for tag in stale_cannon_tags:
            for worker_tag in self._cannon_rush_assignments.pop(tag):
                worker = self.workers.find_by_tag(worker_tag)
                if worker:
                    self.mediator.assign_role(tag=worker_tag, role=UnitRole.GATHERING)

        # Remove dead workers from assignments
        alive_worker_tags = {w.tag for w in self.workers}
        for cannon_tag in list(self._cannon_rush_assignments):
            self._cannon_rush_assignments[cannon_tag] = [
                wt for wt in self._cannon_rush_assignments[cannon_tag]
                if wt in alive_worker_tags
            ]

        # Assign up to 5 workers per cannon under construction
        already_assigned: set[int] = {
            wt for wts in self._cannon_rush_assignments.values() for wt in wts
        }

        for cannon in cannons_under_construction:
            assigned = self._cannon_rush_assignments.setdefault(cannon.tag, [])
            needed = 5 - len(assigned)
            if needed <= 0:
                continue

            available = self.workers.filter(
                lambda w: w.tag not in already_assigned
            )
            # Pick workers closest to the cannon
            sorted_workers = sorted(available, key=lambda w: w.distance_to(cannon.position))
            for worker in sorted_workers[:needed]:
                self.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
                assigned.append(worker.tag)
                already_assigned.add(worker.tag)

        # Micro: attack assigned cannon for each worker
        cannon_by_tag = {c.tag: c for c in cannons_under_construction}
        for cannon_tag, worker_tags in self._cannon_rush_assignments.items():
            cannon = cannon_by_tag.get(cannon_tag)
            if cannon is None:
                continue
            cannon_units = Units([cannon], self)
            for worker_tag in worker_tags:
                worker = self.workers.find_by_tag(worker_tag)
                if not worker:
                    continue
                if worker.is_carrying_resource and self.townhalls:
                    worker.return_resource()
                    continue
                maneuver = CombatManeuver()
                maneuver.add(ShootTargetInRange(unit=worker, targets=cannon_units))
                maneuver.add(AttackTarget(unit=worker, target=cannon))
                self.register_behavior(maneuver)


    async def check_invisible_units(self):
        """Detect cloaked/invisible enemy units (e.g. Dark Templars) and react by
        building Spore Crawlers (static detectors) and an Overseer (mobile detector).

        Uses `is_cloaked and not is_revealed` — the unit is in a cloaked state and
        has not been detected yet, so our units cannot attack it.
        """
        # Once detected, keep building detection every call
        if "Invisible_Unit" in self.enemy_strategy:
            await self.make_spores()
            await self.make_overseer()
            return

        # Scan all visible enemy units for any that are cloaked and undetected
        for enemy in self.enemy_units:
            if enemy.is_cloaked and not enemy.is_revealed:
                await self.chat_send("Tag: Invisible_Unit")
                self.enemy_strategy.append("Invisible_Unit")
                if not self.build_order_runner.build_completed:
                    self.build_order_runner.set_build_completed()
                return

    async def is_late_game_vs_terran(self):
        if self.time > 720:
            if self.late_game == False:
                await self.chat_send("Tag: Late_Game")
                self.enemy_strategy.append("Late_Game")
                self.late_game = True


    async def late_game_vs_terran_protocol(self):
        if not self.late_game:
            return

        drone_count = self.units(UnitID.DRONE).amount
        bases_started = self.townhalls.ready.amount + math.ceil(self.already_pending(UnitID.HATCHERY))

        # Late game can start before 6th base is feasible; release inhibitor by drone target too.
        if bases_started >= 5 or drone_count >= 75:
            self.spawn_inhibitors.discard("late_game_expanding")
            self.late_game_expansion_done = True
        elif not self.late_game_expansion_done:
            self.spawn_inhibitors.add("late_game_expanding")

        macro_plan: MacroPlan = MacroPlan()

        macro_plan.add(ExpansionController(to_count=6, max_pending=2))
        self.register_behavior(macro_plan)

    async def build_hive(self):
        if self.townhalls.of_type({UnitID.HIVE}).ready:
            self.spawn_inhibitors.discard("building_hive")
            return

        # Hive must evolve from a ready Lair and requires a ready Infestation Pit.
        ready_lairs: Units = self.townhalls.of_type({UnitID.LAIR}).ready
        if not ready_lairs or not self.structures(UnitID.INFESTATIONPIT).ready:
            self.spawn_inhibitors.discard("building_hive")
            return

        if self.already_pending(UnitID.HIVE):
            self.spawn_inhibitors.discard("building_hive")
            return

        self.spawn_inhibitors.add("building_hive")

        target_lair: Unit = ready_lairs.closest_to(self.start_location)
        abilities = await self.get_available_abilities(target_lair)
        if AbilityId.UPGRADETOHIVE_HIVE not in abilities:
            return

        if self.can_afford(UnitID.HIVE):
            target_lair(AbilityId.UPGRADETOHIVE_HIVE)


    async def retreat_3rd_and_4rd_overlord(self):
        # The third and fourth overlords are sent to scout whether the enemy has
        # taken their third and fourth bases. Once those bases start construction,
        # the enemy gains vision and these overlords become vulnerable. This
        # function retreats those overlords so they do not die.
        #
        # The first overlord scouts in front of enemy natural and can also die
        # there. If it spots an enemy townhall near its position, retreat early.
        if not self.first_base:
            return

        overlords: Units = self.units(UnitID.OVERLORD)
        if not overlords:
            return

        enemy_townhall_types = {
            UnitID.COMMANDCENTER,
            UnitID.ORBITALCOMMAND,
            UnitID.PLANETARYFORTRESS,
            UnitID.NEXUS,
            UnitID.HATCHERY,
            UnitID.LAIR,
            UnitID.HIVE,
        }
        enemy_townhalls: Units = self.enemy_structures.of_type(enemy_townhall_types)
        if not enemy_townhalls:
            return

        retreat_pos = self.first_base.position.towards(self.game_info.map_center, -15)

        first_overlord: Optional[Unit] = None
        first_overlord_tag = getattr(self.first_overlord, "tag", self.first_overlord)
        if first_overlord_tag:
            first_overlord = overlords.find_by_tag(first_overlord_tag)

        if first_overlord and not self.overlord_retreated:
            should_retreat_first = any(
                base.distance_to(first_overlord.position) < 10
                for base in enemy_townhalls
            )
            if should_retreat_first:
                self.do(first_overlord.move(retreat_pos))
                self.overlord_retreated = True

        third_overlord: Optional[Unit] = None
        fourth_overlord: Optional[Unit] = None

        if self.tag_third_overlord:
            third_overlord = overlords.find_by_tag(self.tag_third_overlord)

        if self.tag_fourth_overlord:
            fourth_overlord = overlords.find_by_tag(self.tag_fourth_overlord)

        for overlord in (third_overlord, fourth_overlord):
            if overlord is None:
                continue

            should_retreat = any(
                base.distance_to(overlord.position) < 10
                for base in enemy_townhalls
            )
            if should_retreat:
                self.do(overlord.move(retreat_pos))

    async def is_marine_rush(self):
        if "Marine_Rush" not in self.enemy_strategy:
            if self.mediator.get_enemy_marine_rush == True:
                await self.chat_send("marine rush")
                self.enemy_strategy.append("Marine_Rush")

    async def is_proxy_stargate(self):
        if "Proxy_Stargate" in self.enemy_strategy:
            return

        enemy_main = self.enemy_start_locations[0]

        for unit in self.enemy_structures:
            if (
                (unit.name == "Stargate" or unit.type_id == UnitID.STARGATE)
                and unit.distance_to(enemy_main) > 20
            ):
                await self.chat_send("Tag: Proxy_Stargate")
                self.enemy_strategy.append("Proxy_Stargate")
                self._apply_proxy_stargate_queen_policy()
                break

    async def build_spores_vs_proxy_stargate(self):
        if "Proxy_Stargate" not in self.enemy_strategy:
            return

        if not self.build_order_runner.build_completed:
            self.build_order_runner.set_build_completed()

        if not self.structures(UnitID.SPAWNINGPOOL).ready:
            return

        if self.first_base is None:
            return

        bases: list[Unit] = [self.first_base]
        if self.second_base is not None and self.second_base.is_ready:
            bases.append(self.second_base)

        def _spores_near_slot(pos: Point2, radius: float = 2.5) -> bool:
            return any(
                s.distance_to(pos) <= radius
                for s in self.structures(UnitID.SPORECRAWLER)
            )

        async def _place_spore_near(slot: Point2) -> Optional[Point2]:
            candidate = slot
            if not self.has_creep(candidate):
                # Pull slightly back toward own base side to find creep nearby.
                candidate = slot.towards(self.start_location, 2)
                if not self.has_creep(candidate):
                    return None

            try:
                placed = await self.find_placement(
                    UnitID.SPORECRAWLER,
                    near=candidate,
                    placement_step=1,
                )
            except Exception:
                placed = None

            result = placed if placed is not None else candidate
            return result if self.has_creep(result) else None

        for base in bases:
            base_pos = base.position

            # Two slots near each base, both facing toward map center.
            to_center = self.game_info.map_center - base_pos
            mag = (to_center.x ** 2 + to_center.y ** 2) ** 0.5 or 1.0
            fwd = Point2((to_center.x / mag, to_center.y / mag))
            perp = Point2((-fwd.y, fwd.x))

            # Slightly farther forward from the hatchery toward map center.
            anchor = Point2((base_pos.x + fwd.x * 8.0, base_pos.y + fwd.y * 5.0))
            slot_left = Point2((anchor.x + perp.x * 2.25, anchor.y + perp.y * 2.25))
            slot_right = Point2((anchor.x - perp.x * 2.25, anchor.y - perp.y * 2.25))

            for slot in (slot_left, slot_right):
                if _spores_near_slot(slot):
                    continue
                if not self.can_afford(UnitID.SPORECRAWLER):
                    return

                build_pos = await _place_spore_near(slot)
                if build_pos is None:
                    continue

                if worker := self.mediator.select_worker(target_position=build_pos):
                    self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                    self.mediator.build_with_specific_worker(
                        worker=worker,
                        structure_type=UnitID.SPORECRAWLER,
                        pos=build_pos,
                        building_purpose=BuildingPurpose.NORMAL_BUILDING,
                    )

    async def make_one_viper(self):
        # Se não tem Hive, não faz nada
        if not self.townhalls.of_type({UnitID.HIVE}).ready:
            self.spawn_inhibitors.discard("making_viper")
            return

        # Se já tem viper ou está treinando uma, não precisa fazer mais
        if self.units(UnitID.VIPER) or self.already_pending(UnitID.VIPER):
            self.spawn_inhibitors.discard("making_viper")
            return

        # Se pode gastar recursos para fazer uma viper, faz via larva
        if self.can_afford(UnitID.VIPER):
            for larva in self.units(UnitID.LARVA):
                larva.train(UnitID.VIPER)
                self.spawn_inhibitors.discard("making_viper")
                break
        else:
            # Se não pode gastar, adiciona o inhibitor para bloquear spawn de outras unidades
            self.spawn_inhibitors.add("making_viper")
#_______________________________________________________________________________________________________________________
#          DEBUG TOOL
#_______________________________________________________________________________________________________________________

    async def debug_tool(self):
        current_time = time.time()
        if current_time - self.last_debug_time >= 1:  # Se passou mais de um segundo
            print("Time: ", self.time)
            #print(self.mediator.get_all_enemy)
            #print("Enemy Race: ", self.EnemyRace)
            #print("Second Base: ", self.second_base)
            print("Enemy Strategy: ", self.enemy_strategy)
            #print("Random Race Discovered: ", self.random_race_discovered)
            #print("Creep Queen Policy: ", self.creep_queen_policy)
            #print("RallyPointSet: ", self.rally_point_set)
            #print("nydus_position: ", self.mediator.get_primary_nydus_own_main)
            #print("Enemy Units: ", self.enemy_units)
            #print("Enemy Structures: ", self.enemy_structures)
            print("Spawn Inibitors:", self.spawn_inhibitors)
            #print("Second Overlord: ", self.tag_second_overlord)
            #print("Mutalisk targets:", self.mutalisk_targets)
            #print("Behind mineral positions: ", self.mediator.get_behind_mineral_positions(th_pos=self.first_base.position))
            #print("Enemy Start Location: ", self.enemy_start_locations[0])
            #print("Build Completed: ", self.build_order_runner.build_completed)
            #print("Scout Targets", self.scout_targets)
            #print("Max creep queens:", self.max_creep_queens)
            #print("Creep queen tags:", self.creep_queen_tags)
            #print("Other Queens:", self.other_queen_tags)
            #print("Enemies on creep:", self.enemies_on_creep)
            #print("worker rush:", self.mediator.get_enemy_worker_rushed)
            #print("My Overlords:", self.my_overlords)
            #print("My roaches:", self.my_roaches)
            #print("FirstBase: ", self.first_base)
            #print("SecondBase: ", self.second_base)
            #print("Enemy Widow Mines: ", self.enemy_widow_mines)
            self.last_debug_time = current_time  # Atualizar a última vez que a ferramenta de debug foi chamada


#_______________________________________________________________________________________________________________________
#          ON UNIT TOOK DAMAGE
#_______________________________________________________________________________________________________________________

    # If the building is attacked and is not complete, cancel the construction

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)


        # If the building is attacked and is not complete, cancel the construction
        compare_health: float = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health and unit.is_structure:
            unit(AbilityId.CANCEL_BUILDINPROGRESS)


        if unit.type_id == UnitID.ROACH:
            self.is_roach_attacking = True
             

#_______________________________________________________________________________________________________________________
#          ON UNIT DESTROYED
#_______________________________________________________________________________________________________________________
    async def on_unit_destroyed(self, unit_tag: int) -> None:
        await super(MyBot, self).on_unit_destroyed(unit_tag)
    
        # Verifica se o primeiro overlord morreu antes de 1 minuto
        if hasattr(self, "first_overlord") and self.first_overlord is not None:
            if unit_tag == self.first_overlord.tag and self.time < 160:
                await self.chat_send("Tag: First_Overlord_Killed")



        # checks if unit is a queen or th, library then handles appropriately
        self.queens.remove_unit(unit_tag)

        if unit_tag in self.creep_queen_tags:
            self.creep_queen_tags.remove(unit_tag)

        # Se a segunda base foi destruída, reseta o rally point para a primeira base
        if self.second_base is not None and unit_tag == self.second_base.tag:
            self.second_base = None
            rally_point = self.first_base.position.towards(self.game_info.map_center, 6)
            for hatchery in self.structures(UnitID.HATCHERY).ready:
                self.do(hatchery(AbilityId.RALLY_HATCHERY_UNITS, rally_point))
            





#_______________________________________________________________________________________________________________________
#          ON UNIT CREATED
#_______________________________________________________________________________________________________________________


    async def on_unit_created(self, unit: Unit) -> None:
        """
        Can use burnysc2 hooks as usual, just add a call to the
        parent method before your own logic.
        """
        await super(MyBot, self).on_unit_created(unit)

        # Adicionar Overlords ao dicionário self.my_overlords
        if unit.type_id == UnitID.OVERLORD:
            self.my_overlords[unit.tag] = unit

        # Adicionar Roaches ao dicionário self.my_roaches
        if unit.type_id == UnitID.ROACH:
            self.my_roaches[unit.tag] = unit




        # assign our forces ATTACKING by default
        if unit.type_id not in WORKER_TYPES and unit.type_id not in {
            UnitID.QUEEN,
            UnitID.MULE,
            UnitID.OVERLORD,
            UnitID.CHANGELING,
        }:
            # here we are making a request to an ares manager via the mediator
            # See https://aressc2.github.io/ares-sc2/api_reference/manager_mediator.html
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)




        # Exemplo para a segunda base:
        if unit.type_id == UnitID.OVERLORD and self.units(UnitID.OVERLORD).amount == 2:
            self.tag_second_overlord = unit.tag
            if "Cannon_Rush" in self.enemy_strategy:
                target = self.mediator.get_own_nat
            elif "Magannatha AIE" in self.game_info.map_name:
                enemy_third = self.mediator.get_enemy_third
                target = enemy_third.towards(self.enemy_start_locations[0], 13)
            else:
                target = self.mediator.get_primary_nydus_enemy_main
            self.do(unit.move(target))
            await self.chat_send("Tag: Version_260720")
        
        # Exemplo para a terceira base:
        if unit.type_id == UnitID.OVERLORD and self.units(UnitID.OVERLORD).amount == 3:
            self.tag_third_overlord = unit.tag
            enemy_third = self.mediator.get_enemy_third
            target = enemy_third.position.towards(self.game_info.map_center, 9)
            self.do(unit.move(target))
        
        # Exemplo para a quarta base:
        if unit.type_id == UnitID.OVERLORD and self.units(UnitID.OVERLORD).amount == 4:
            self.tag_fourth_overlord = unit.tag
            enemy_fourth = self.mediator.get_enemy_fourth
            target = enemy_fourth.position.towards(self.game_info.map_center, 9)
            self.do(unit.move(target))


        # For the third Overlord and beyond, send them behind the first base
        elif unit.type_id == UnitID.OVERLORD and self.units(UnitID.OVERLORD).amount >= 5:

            target = self.first_base.position.towards(self.game_info.map_center, -15)  # Get a position behind of the first base
            self.do(unit.move(target))

#_______________________________________________________________________________________________________________________
#          ON BUILDING CONSTRUCTION COMPLETE
#_______________________________________________________________________________________________________________________


    async def on_building_construction_complete(self, unit: Unit) -> None:
        await super(MyBot, self).on_building_construction_complete(unit)


        #when the second base is built, set the rally point to the second base
        if unit.type_id == UnitID.HATCHERY:
            self.rally_point_set = True  
            bases = self.structures(UnitID.HATCHERY).ready
            if bases.amount == 2:
                for base in bases:
                    if base.tag != self.first_base.tag:
                        self.second_base = base
                        break

            if self.second_base is not None:         
                rally_point = self.second_base.position.towards(self.game_info.map_center, 6)                          

                for hatcherys in self.structures(UnitID.HATCHERY).ready:
                    self.do(hatcherys(AbilityId.RALLY_HATCHERY_UNITS, rally_point))




#_______________________________________________________________________________________________________________________
#          DEF MACRO
#_______________________________________________________________________________________________________________________

    def _macro(self) -> None:

        # MAKE SUPPLY
        # ares-sc2 AutoSupply
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.auto_supply.AutoSupply
        # Always register AutoSupply so overlords are made even if the build order
        # is blocked mid-execution (e.g. waiting for RoachWarren during CannonRush).
        self.register_behavior(AutoSupply(base_location=self.start_location))



        # MINE
        # ares-sc2 Mining behavior
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.mining.Mining
        #
        # Reutilizamos self.mining_behavior (criado no on_start) em vez de instanciar
        # Mining() a cada frame. Isso preserva os caches internos da classe:
        #   - locked_action_tags: evita spam de comandos repetidos nos workers
        #   - safe_long_distance_mineral_fields: lista de patches seguros para mineração
        #     a longa distância — calculada apenas quando None e muito custosa (O(n))
        # Apenas atualizamos workers_per_gas caso o valor tenha mudado em tempo de jogo.
        if self.speedMiningOn == True:
            self.mining_behavior.workers_per_gas = self.workers_for_gas
            self.register_behavior(self.mining_behavior)




#_______________________________________________________________________________________________________________________
        # BUILD ARMY
        # ares-sc2 SpawnController
#_______________________________________________________________________________________________________________________


        if not self.spawn_inhibitors:


            if self.EnemyRace == Race.Terran:
                if "Flying_Structures" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_MUTAROACH[self.race]))
                elif "Mass_Banshee" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_MUTAROACH[self.race]))    
                elif "Battlecruiser" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_ROACHCORRUPTOR[self.race]))
                else:
                    self.register_behavior(SpawnController(ARMY_COMP_ROACHINFESTOR[self.race]))
            
            elif self.EnemyRace == Race.Protoss:
                if "2_Proxy_Gateway" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                elif "Cannon_Rush" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                elif "Proxy_Stargate" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_LING[self.race]))
                elif "Protoss_Agressive" in self.enemy_strategy and not "Proxy_Stargate" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                else:
                    self.register_behavior(SpawnController(ARMY_COMP_LING[self.race]))
            
            elif self.EnemyRace == Race.Zerg:
                if "Worker_Rush" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_LING[self.race]))
                else:    
                    self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
            
            elif self.EnemyRace == Race.Random:
                if "Worker_Rush" in self.enemy_strategy:
                    self.register_behavior(SpawnController(ARMY_COMP_LING[self.race]))
                else:
                    if "Random_Protoss" in self.enemy_strategy:
                        if "2_Proxy_Gateway" in self.enemy_strategy:
                            self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                        elif "Cannon_Rush" in self.enemy_strategy:
                            self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                        elif "Protoss_Agressive" in self.enemy_strategy:
                            self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))
                        else:
                            self.register_behavior(SpawnController(ARMY_COMP_LING[self.race]))

                    else:
                        self.register_behavior(SpawnController(ARMY_COMP_ROACH[self.race]))

        # see also `ProductionController` for ongoing generic production, not needed here
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.spawn_controller.ProductionController




        self._zerg_specific_macro()

#_______________________________________________________________________________________________________________________
#          RAVAGER BILE FALLBACK
#_______________________________________________________________________________________________________________________

    async def _force_ravager_bile_each_frame(self) -> None:
        ravagers: Units = self.units(UnitID.RAVAGER)
        if not ravagers:
            return

        BILE_RANGE: float = 9.0
        BILE_PRIORITY_TYPES: tuple[UnitID, ...] = (
            UnitID.SIEGETANKSIEGED,
            UnitID.LIBERATORAG,
            UnitID.MEDIVAC,
        )

        def _priority_tier(u: Unit) -> int:
            t = u.type_id
            if t == UnitID.SIEGETANKSIEGED:
                return 0
            if t == UnitID.LIBERATORAG:
                return 1
            if t == UnitID.MEDIVAC:
                return 2
            return 99

        # Visíveis no frame atual: unidades + estruturas.
        enemy_candidates: Units = (
            self.enemy_units.filter(lambda u: not u.is_memory)
            + self.enemy_structures.filter(lambda u: not u.is_memory)
        )
        if not enemy_candidates:
            return

        for ravager in ravagers:
            # Não sobrescreve um bile já iniciado.
            if any(order.ability.id == AbilityId.EFFECT_CORROSIVEBILE for order in ravager.orders):
                continue

            in_bile_range: list[Unit] = [
                u
                for u in enemy_candidates
                if cy_distance_to(ravager.position, u.position) <= BILE_RANGE
            ]
            if not in_bile_range:
                continue

            priority_targets: list[Unit] = [
                u for u in in_bile_range if u.type_id in BILE_PRIORITY_TYPES
            ]

            if priority_targets:
                best_bile: Unit = min(
                    priority_targets,
                    key=lambda u: (_priority_tier(u), cy_distance_to(ravager.position, u.position)),
                )
            else:
                # Fallback: se não houver alvo da lista, bile em qualquer inimigo no range.
                best_bile = min(
                    in_bile_range,
                    key=lambda u: cy_distance_to(ravager.position, u.position),
                )

            cached_abilities = await self.get_available_abilities(ravager)
            if await self.can_cast(
                ravager,
                AbilityId.EFFECT_CORROSIVEBILE,
                target=best_bile.position,
                cached_abilities_of_unit=cached_abilities,
            ):
                ravager(AbilityId.EFFECT_CORROSIVEBILE, best_bile.position)

#_______________________________________________________________________________________________________________________
#          DEF _MICRO
#_______________________________________________________________________________________________________________________

    def _micro(self, forces: Units) -> None:
        # make a fast batch distance query to enemy units for all our units
        # key: unit tag, value: units in range of that unit tag
        # https://aressc2.github.io/ares-sc2/api_reference/manager_mediator.html#ares.managers.manager_mediator.ManagerMediator.get_units_in_range
        # as zerg we will only interact with ground enemy, else we should get all enemy
        query_type: UnitTreeQueryType = (
            UnitTreeQueryType.EnemyGround
            if self.race == Race.Zerg
            else UnitTreeQueryType.AllEnemy
        )
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=forces,
            distances=15,
            query_tree=query_type,
            return_as_dict=True,
        )

        # get a ground grid to path on, this already contains enemy influence
        grid: np.ndarray = self.mediator.get_ground_grid

        # make a single call to self.attack_target property
        # otherwise it keep calculating for every unit
        target: Point2 = self.attack_target

        # Atualizar o alvo se houver inimigos na creep
        if self.enemies_on_creep:
            first_enemy_on_creep = next(iter(self.enemies_on_creep.values()))
            target = first_enemy_on_creep.position

        # Pre-compute shared mutalisk target (once per step)
        def _muta_priority(u: Unit) -> int:
            if u.type_id == UnitID.BANSHEE:
                return 0
            if u.type_id in {UnitID.VIKINGFIGHTER, UnitID.VIKINGASSAULT}:
                return 1
            if u.type_id == UnitID.MEDIVAC:
                return 2
            if u.type_id in {UnitID.SIEGETANK, UnitID.SIEGETANKSIEGED}:
                return 3
            if u.type_id == UnitID.SCV:
                return 5
            if u.is_flying and u.type_id in ALL_STRUCTURES:
                return 6
            return 4

        _muta_shared_target: Optional[Unit] = None
        _all_enemy_for_muta: Optional[Units] = None
        _muta_units: Units = forces.filter(lambda u: u.type_id == UnitID.MUTALISK)
        if _muta_units:
            _combined = (
                self.enemy_units.filter(lambda u: not u.is_memory)
                + self.enemy_structures.filter(lambda u: not u.is_memory)
            )
            if _combined:
                _all_enemy_for_muta = _combined
                _enemy_by_tag: dict[int, Unit] = {u.tag: u for u in _combined}
                _current: Optional[Unit] = _enemy_by_tag.get(self.mutalisk_targets.get(0))
                if _current is None:
                    _best = sorted(
                        _combined,
                        key=lambda u: (_muta_priority(u), u.distance_to(_muta_units.center)),
                    )
                    if _best:
                        _current = _best[0]
                        self.mutalisk_targets[0] = _current.tag
                else:
                    _cp = _muta_priority(_current)
                    if _cp > 0:
                        _better = [u for u in _combined if _muta_priority(u) < _cp]
                        if _better:
                            _bp = min(_muta_priority(u) for u in _better)
                            _top = [u for u in _better if _muta_priority(u) == _bp]
                            _current = min(_top, key=lambda u: u.distance_to(_muta_units.center))
                            self.mutalisk_targets[0] = _current.tag
                _muta_shared_target = _current

        # Bile target: prioriza SiegeTank se visível, senão usa attack_target
        bile_target: Point2 = self.attack_target
        for _u in self.enemy_units:
            if _u.name == 'SiegeTank':
                bile_target = _u.position
                break

        # use `ares-sc2` combat maneuver system
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/combat_behaviors.html
        for unit in forces:
            """
            Set up a new CombatManeuver, idea here is to orchestrate your micro
            by stacking behaviors in order of priority. If a behavior executes
            then all other behaviors will be ignored for this step.
            """

            attacking_maneuver: CombatManeuver = CombatManeuver()
            # we already calculated close enemies, use unit tag to retrieve them
            all_close: Units = near_enemy[unit.tag].filter(
                lambda u: not u.is_memory and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            )
            only_enemy_units: Units = all_close.filter(
                lambda u: u.type_id not in ALL_STRUCTURES
            )

            if unit.type_id in [UnitID.ROACH, UnitID.ROACHBURROWED]:
                # only roaches can burrow
                burrow_behavior: CombatManeuver = self.burrow_behavior(unit)
                attacking_maneuver.add(burrow_behavior)

#_______________________________________________________________________________________________________________________
#          CORRUPTOR
#_______________________________________________________________________________________________________________________
            if unit.type_id == UnitID.CORRUPTOR:
                CORRUPTOR_AIR_PRIORITY: list[UnitID] = [
                    UnitID.BATTLECRUISER,
                    UnitID.VIKINGFIGHTER,
                    UnitID.BANSHEE,
                    UnitID.MEDIVAC,
                ]
                # Query all enemy units globally — near_enemy uses EnemyGround and misses air
                all_enemy_air: Units = self.enemy_units.filter(
                    lambda u: u.is_flying and not u.is_memory
                )
                if all_enemy_air:
                    # Find the highest-priority air target on the entire map
                    global_air_target: Optional[Unit] = None
                    for p_type in CORRUPTOR_AIR_PRIORITY:
                        candidates = all_enemy_air.of_type(p_type)
                        if candidates:
                            global_air_target = min(
                                candidates, key=lambda u: u.distance_to(unit)
                            )
                            break
                    if not global_air_target:
                        global_air_target = min(
                            all_enemy_air, key=lambda u: u.distance_to(unit)
                        )

                    if in_attack_range := cy_in_attack_range(unit, all_enemy_air):
                        # Shoot highest-priority type already in weapon range
                        priority_in_range = None
                        for p_type in CORRUPTOR_AIR_PRIORITY:
                            candidates = [u for u in in_attack_range if u.type_id == p_type]
                            if candidates:
                                priority_in_range = candidates
                                break
                        targets = priority_in_range if priority_in_range else in_attack_range
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=targets)
                        )
                    else:
                        # Path to the priority air target anywhere on the map
                        attacking_maneuver.add(
                            PathUnitToTarget(unit=unit, grid=grid, target=global_air_target.position)
                        )
                        attacking_maneuver.add(
                            AMove(unit=unit, target=global_air_target.position)
                        )
                self.register_behavior(attacking_maneuver)
                continue

#_______________________________________________________________________________________________________________________
#          MUTALISK
#_______________________________________________________________________________________________________________________
            if unit.type_id == UnitID.MUTALISK:
                air_grid: np.ndarray = self.mediator.get_air_grid
                # Retreat logic mirrored from return_to_base — must run inside _micro
                # so the registered behavior isn't overridden by a later unit.move() call.
                if self._commenced_attack and self.get_total_supply(forces) < 0.5 * self._begin_attack_at_supply:
                    _bases = self.structures(UnitID.HATCHERY).ready
                    _base_ref = (
                        self.second_base
                        if _bases.amount >= 2 and self.second_base is not None
                        else self.first_base
                    )
                    _base_under_attack = any(
                        e.distance_to(_base_ref.position) < 18 or self.has_creep(e.position)
                        for e in self.enemy_units
                    )
                    if not _base_under_attack:
                        _retreat_pos = _base_ref.position.towards(self.game_info.map_center, 6)
                        attacking_maneuver.add(PathUnitToTarget(unit=unit, grid=air_grid, target=_retreat_pos))
                        attacking_maneuver.add(AMove(unit=unit, target=_retreat_pos))
                        self.register_behavior(attacking_maneuver)
                        continue
                if unit.health_percentage < 0.6:
                    nearest_base = (
                        min(self.townhalls.ready, key=lambda b: b.distance_to(unit)).position
                        if self.townhalls.ready else self.start_location
                    )
                    attacking_maneuver.add(PathUnitToTarget(unit=unit, grid=air_grid, target=nearest_base))
                    attacking_maneuver.add(AMove(unit=unit, target=nearest_base))
                    self.register_behavior(attacking_maneuver)
                    continue
                # Flee from Missile Turrets — they deal massive damage and mutas can't hit them
                TURRET_RANGE: float = 8.0
                nearby_turrets = [
                    s for s in self.enemy_structures
                    if s.type_id == UnitID.MISSILETURRET
                    and not s.is_memory
                    and cy_distance_to(unit.position, s.position) <= TURRET_RANGE
                ]
                if nearby_turrets:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=air_grid))
                    self.register_behavior(attacking_maneuver)
                    continue
                if not self._commenced_attack:
                    # Modo defensivo: só ataca unidades voadoras sobre a creep.
                    # Não avança em direção a alvos terrestres nem à base inimiga.
                    enemy_air = self.enemy_units.filter(
                        lambda u: u.is_flying and not u.is_memory
                        and self.has_creep(u.position)
                    )
                    if enemy_air:
                        if in_attack_range := cy_in_attack_range(unit, enemy_air):
                            attacking_maneuver.add(
                                ShootTargetInRange(unit=unit, targets=in_attack_range)
                            )
                        closest_air = min(enemy_air, key=lambda u: u.distance_to(unit))
                        attacking_maneuver.add(AMove(unit=unit, target=closest_air.position))
                    self.register_behavior(attacking_maneuver)
                    continue
                # Modo ataque: usa o alvo compartilhado por prioridade
                if _muta_shared_target is None or _all_enemy_for_muta is None:
                    attacking_maneuver.add(
                        PathUnitToTarget(unit=unit, grid=grid, target=target)
                    )
                    attacking_maneuver.add(AMove(unit=unit, target=target))
                else:
                    if in_attack_range := cy_in_attack_range(unit, _all_enemy_for_muta):
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=in_attack_range)
                        )
                    attacking_maneuver.add(
                        AMove(unit=unit, target=_muta_shared_target.position)
                    )
                self.register_behavior(attacking_maneuver)
                continue

#_______________________________________________________________________________________________________________________
#          RAVAGER
#_______________________________________________________________________________________________________________________

            if unit.type_id in [UnitID.RAVAGER]:
                # Controle manual de cooldown (7s = ~157 frames).
                # unit.abilities não é populado de forma síncrona no python-sc2,
                # por isso rastreamos o cooldown manualmente.
                if not hasattr(self, "_bile_cd"):
                    self._bile_cd: dict[int, int] = {}

                BILE_RANGE: int = 9
                BILE_AIR_TYPES: set[UnitID] = {UnitID.LIBERATORAG, UnitID.MEDIVAC}

                if self._bile_cd.get(unit.tag, 0) <= self.state.game_loop:
                    air_bile: list[Unit] = [
                        u for u in self.enemy_units
                        if u.type_id in BILE_AIR_TYPES
                        and not u.is_memory
                        and cy_distance_to(unit.position, u.position) <= BILE_RANGE
                    ]
                    ground_bile: list[Unit] = [
                        u for u in all_close
                        if u.type_id != UnitID.BANSHEE
                    ]

                    def _bile_tier(u: Unit) -> int:
                        t = u.type_id
                        if t == UnitID.SIEGETANKSIEGED: return 0
                        if t == UnitID.LIBERATORAG:     return 1
                        if t == UnitID.MEDIVAC:         return 2
                        if t in WORKER_TYPES:           return 4
                        if t in ALL_STRUCTURES:         return 5
                        return 3

                    all_bile_candidates: list[Unit] = air_bile + ground_bile
                    if all_bile_candidates:
                        best_bile = min(
                            all_bile_candidates,
                            key=lambda u: (_bile_tier(u), cy_distance_to(unit.position, u.position)),
                        )
                        unit(AbilityId.EFFECT_CORROSIVEBILE, best_bile.position)
                        self._bile_cd[unit.tag] = self.state.game_loop + int(22.4 * 7.0) + 6
                        # Bile disparada: não registrar mais nada neste frame
                        # para evitar que outro comando sobrescreva a habilidade.
                        continue

                # Bile em cooldown: comportamento normal de ataque/movimento
                if all_close:
                    if in_attack_range := cy_in_attack_range(unit, only_enemy_units):
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=in_attack_range)
                        )
                    elif in_attack_range := cy_in_attack_range(unit, all_close):
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=in_attack_range)
                        )
                    enemy_target: Unit = cy_pick_enemy_target(all_close)
                    attacking_maneuver.add(
                        StutterUnitBack(unit=unit, target=enemy_target, grid=grid)
                    )
                else:
                    # Sem inimigos terrestres: avança para alvo aéreo ou alvo geral
                    air_targets: list[Unit] = [
                        u for u in self.enemy_units
                        if u.type_id in {UnitID.LIBERATORAG, UnitID.MEDIVAC}
                        and not u.is_memory
                    ]
                    move_target: Point2 = (
                        min(air_targets, key=lambda u: u.distance_to(unit)).position
                        if air_targets else target
                    )
                    attacking_maneuver.add(
                        PathUnitToTarget(unit=unit, grid=grid, target=move_target)
                    )
                    attacking_maneuver.add(AMove(unit=unit, target=move_target))
                self.register_behavior(attacking_maneuver)
                continue

            # enemy around, engagement control
            if all_close:
                # ares's cython version of `cy_in_attack_range` is approximately 4
                # times speedup vs burnysc2's `all_close.in_attack_range_of`

                # idea here is to attack anything in range if weapon is ready
                # check for enemy units first



#_______________________________________________________________________________________________________________________
#          ROACH
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.ROACH]:
                    if in_attack_range := cy_in_attack_range(unit, only_enemy_units):
                        # `ShootTargetInRange` will check weapon is ready
                        # otherwise it will not execute
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=in_attack_range)
                        )
                    # then enemy structures
                    elif in_attack_range := cy_in_attack_range(unit, all_close):
                        attacking_maneuver.add(
                            ShootTargetInRange(unit=unit, targets=in_attack_range)
                        )

                    enemy_target: Unit = cy_pick_enemy_target(all_close)

                    # low shield, keep protoss units safe
                    if self.race == Race.Protoss and unit.shield_percentage < 0.3:
                        attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))

                    else:
                        attacking_maneuver.add(
                            StutterUnitBack(unit=unit, target=enemy_target, grid=grid)
                        )

#_______________________________________________________________________________________________________________________
#          ZERGLING
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.ZERGLING]:
                    if self.units(UnitID.ROACH).amount > 0:
                        if self.is_roach_attacking:
                            attacking_maneuver.add(AMove(unit=unit, target=target))
                        
                        else:
                            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                
                    else:
                        attacking_maneuver.add(AMove(unit=unit, target=target))


#_______________________________________________________________________________________________________________________
#          ROACH BURROWED
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.ROACHBURROWED]:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))



#_______________________________________________________________________________________________________________________
#          INFESTOR
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.INFESTOR]:
                    # Cooldown global: nenhum infestor pode lançar enquanto o timer estiver ativo
                    fungal_global_ready: bool = getattr(self, "_fungal_global_cd", 0) <= self.state.game_loop
                    if not fungal_global_ready:
                        attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                        self.register_behavior(attacking_maneuver)
                        continue
                    fungal_targets: list[Unit] = [
                        u for u in only_enemy_units
                        if u.type_id != UnitID.RAVEN
                    ]
                    if fungal_targets and unit.energy >= 75:
                        best_pos: Point2 | None = None
                        best_count: int = 0
                        for candidate in fungal_targets:
                            count = sum(
                                1 for u in fungal_targets
                                if cy_distance_to(candidate.position, u.position) <= 2.0
                            )
                            if count > best_count:
                                best_count = count
                                best_pos = candidate.position
                        if best_pos and best_count >= 3:
                            unit(AbilityId.FUNGALGROWTH_FUNGALGROWTH, best_pos)
                            # Cooldown global de 3 segundos (~67 frames a 22.4 fps)
                            self._fungal_global_cd = self.state.game_loop + int(22.4 * 3.0) + 4
                            self.register_behavior(attacking_maneuver)
                            continue
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                    self.register_behavior(attacking_maneuver)
                    continue
#_______________________________________________________________________________________________________________________
#          INFESTOR BURROWED
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.INFESTORBURROWED]:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                    self.register_behavior(attacking_maneuver)
                    continue

#_______________________________________________________________________________________________________________________
#          VIPER
#_______________________________________________________________________________________________________________________

                if unit.type_id in [UnitID.VIPER]:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                    self.register_behavior(attacking_maneuver)
                    continue
#_______________________________________________________________________________________________________________________
#          OTHER UNITS
#_______________________________________________________________________________________________________________________

                attacking_maneuver.add(AMove(unit=unit, target=target))

                    
            # no enemy around, path to the attack target
            else:
                attacking_maneuver.add(
                    PathUnitToTarget(unit=unit, grid=grid, target=target)
                )
                attacking_maneuver.add(AMove(unit=unit, target=target))

            # DON'T FORGET TO REGISTER OUR COMBAT MANEUVER!!
            self.register_behavior(attacking_maneuver)

    def burrow_behavior(self, roach: Unit) -> CombatManeuver:
        """
        Burrow or unburrow roach
        """
        burrow_maneuver: CombatManeuver = CombatManeuver()
        if roach.is_burrowed and roach.health_percentage > self.UNBURROW_AT_HEALTH_PERC:
            burrow_maneuver.add(UseAbility(AbilityId.BURROWUP_ROACH, roach, None))
        elif (
            not roach.is_burrowed
            and roach.health_percentage <= self.BURROW_AT_HEALTH_PERC
        ):
            burrow_maneuver.add(UseAbility(AbilityId.BURROWDOWN_ROACH, roach, None))

        return burrow_maneuver



#_______________________________________________________________________________________________________________________
#          ZERG MACRO
#_______________________________________________________________________________________________________________________


    def _zerg_specific_macro(self) -> None:
        if self.EnemyRace == Race.Terran:  
            
            if (not self.already_pending_upgrade(UpgradeId.BURROW)):
                self.research(UpgradeId.BURROW)

            if (not self.already_pending_upgrade(UpgradeId.TUNNELINGCLAWS)):
                self.research(UpgradeId.TUNNELINGCLAWS)



        if self.EnemyRace == Race.Protoss:
            if "2_Base_Protoss" in self.enemy_strategy:
                if (not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)):
                    self.research(UpgradeId.ZERGLINGMOVEMENTSPEED)       
            if "Protoss_Agressive" in self.enemy_strategy:
                if (not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)):
                    self.research(UpgradeId.BURROW)

        if self.EnemyRace == Race.Zerg:  
            
            if (not self.already_pending_upgrade(UpgradeId.BURROW)):
                self.research(UpgradeId.BURROW)

            if (not self.already_pending_upgrade(UpgradeId.TUNNELINGCLAWS)):
                self.research(UpgradeId.TUNNELINGCLAWS)


        if self.EnemyRace == Race.Random:
            if "Random_Protoss" in self.enemy_strategy:
                if (not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)):
                    self.research(UpgradeId.ZERGLINGMOVEMENTSPEED)

            else:
                if (not self.already_pending_upgrade(UpgradeId.BURROW)):
                    self.research(UpgradeId.BURROW)


    """
        for queen in self.mediator.get_own_army_dict[UnitID.QUEEN]:
            if queen.energy >= 25 and self.townhalls:
                queen(AbilityId.EFFECT_INJECTLARVA, self.townhalls[0])

    Can use `python-sc2` hooks as usual, but make a call the inherited method in the superclass
    Examples:

    # async def on_end(self, game_result: Result) -> None:
    #     await super(MyBot, self).on_end(game_result)
    #
    #     # custom on_end logic here ...
    #

    #     # custom on_building_construction_complete logic here ...
    #
    # async def on_unit_destroyed(self, unit_tag: int) -> None:
    #     await super(MyBot, self).on_unit_destroyed(unit_tag)
    #
    #     # custom on_unit_destroyed logic here ...
    #
    # async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
    #     await super(MyBot, self).on_unit_took_damage(unit, amount_damage_taken)
    #
    #     # custom on_unit_took_damage logic here ...


    async def build_zerglings(self):
        if (self.minerals/ self.vespene + 1) > 5 and self.minerals > 1000:
            for larva in self.units(UnitID.LARVA):
                # Check if we can afford a Zergling and have enough supply
                if self.can_afford(UnitID.ZERGLING) and self.supply_left > 0:
                    # If we can, train a Zergling
                    self.do(larva.train(UnitID.ZERGLING))
    """