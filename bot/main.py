"""
---------------------------
BOT CLICADINHA
---------------------------

Made of with Ares Random Example Bot

"""


from itertools import cycle
from typing import Optional

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
)
from ares.behaviors.macro import AutoSupply, Mining, SpawnController
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole, UnitTreeQueryType, BuildingPurpose
from cython_extensions import cy_closest_to, cy_in_attack_range, cy_pick_enemy_target
from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
import time


#_______________________________________________________________________________________________________________________
#          ARMY COMPOSITION
#_______________________________________________________________________________________________________________________


ARMY_COMP_VS_TERRAN: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ZERGLING: {"proportion": 0.2, "priority": 0},
        UnitID.HYDRALISK: {"proportion": 0.8, "priority": 0},
    }
}
# this will be used for ares SpawnController behavior
ARMY_COMP: dict[Race, dict] = {
    Race.Zerg: {
        UnitID.ROACH: {"proportion": 1.0, "priority": 0},
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
    BURROW_AT_HEALTH_PERC: float = 0.3
    UNBURROW_AT_HEALTH_PERC: float = 0.9
    last_debug_time = 0
    

    def __init__(self, game_step_override: Optional[int] = None):
        

        """Initiate custom bot

        Parameters
        ----------
        game_step_override :
            If provided, set the game_step to this value regardless of how it was
            specified elsewhere
        """
        super().__init__(game_step_override)
        self.tag_worker_build_2nd_base = 0    

        self._commenced_attack: bool = False


        

    @property
    def attack_target(self) -> Point2:
        if self.enemy_structures:
            # using a faster cython alternative here, see docs for all available functions
            # https://aressc2.github.io/ares-sc2/api_reference/cython_extensions/index.html
            return cy_closest_to(self.start_location, self.enemy_structures).position
        # not seen anything in early game, just head to enemy spawn
        elif self.time < 240.0:
            return self.enemy_start_locations[0]
        # else search the map
        else:
            # cycle through expansion locations
            if self.is_visible(self.current_base_target):
                self.current_base_target = next(self.expansions_generator)

            return self.current_base_target
        
#_______________________________________________________________________________________________________________________
#          ON START
#_______________________________________________________________________________________________________________________

    async def on_start(self) -> None:
        """
        Can use burnysc2 hooks as usual, just add a call to the
        parent method before your own logic.
        """
        await super(MyBot, self).on_start()

        self.EnemyRace = self.enemy_race  
        self.rally_point_set = False
        self.first_base = self.townhalls.first
        self.second_base = None
        self.worker_scout_tag = 0
        self.guess_strategy = "No strategy detected"

        self.current_base_target = self.enemy_start_locations[0]
        self.expansions_generator = cycle(
            [pos for pos in self.expansion_locations_list]
        )

        if self.EnemyRace == Race.Terran:
            self._begin_attack_at_supply = 20
        
        if self.EnemyRace == Race.Protoss:
            self._begin_attack_at_supply = 4
        
        else:
            self._begin_attack_at_supply = 14

        # Send Overlord to scout on the second base
        await self.send_overlord_to_scout()

#_______________________________________________________________________________________________________________________
#          SEND OVERLORD TO SCOUT
#_______________________________________________________________________________________________________________________


    async def send_overlord_to_scout(self):
        # Select the first Overlord
        overlord = self.units(UnitID.OVERLORD).first
    
        enemy_natural_location = self.mediator.get_enemy_nat
    
        # Get the enemy's start location
        #enemy_natural_location = self.mediator.get_enemy_nat
        target = self.mediator.get_closest_overlord_spot(from_pos=enemy_natural_location)
    
        # Send the Overlord to the new position
        self.do(overlord.move(target))


#_______________________________________________________________________________________________________________________
#          ON STEP
#_______________________________________________________________________________________________________________________

    async def on_step(self, iteration: int) -> None:
        await super(MyBot, self).on_step(iteration)

        #await self.debug_tool()

        self._macro()
        await self.inject_larva()

        # https://aressc2.github.io/ares-sc2/api_reference/manager_mediator.html#ares.managers.manager_mediator.ManagerMediator.get_units_from_role
        # see `self.on_unit_created` where we originally assigned units ATTACKING role
        forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        if self._commenced_attack:
            self._micro(forces)

        elif self.get_total_supply(forces) >= self._begin_attack_at_supply:
            self._commenced_attack = True
    
        if self.EnemyRace == Race.Terran:
            await self.build_queens()
            await self.build_next_base()

        if self.EnemyRace == Race.Protoss:
            await self.build_queens()
            await self.build_next_base()

            # If we don't have enough army, stop attacking and build more units
            if self.get_total_supply(forces) <= self._begin_attack_at_supply:
                self._commenced_attack = False

    async def build_queens(self):
        # Loop over each townhall (Hatchery, Lair, or Hive)
        for th in self.townhalls.ready:
            # Check if there's already a queen for this townhall
            queens = self.units(UnitID.QUEEN).closer_than(5, th)
            # Check if the townhall is currently training a queen
            is_training_queen = len(th.orders) > 0
            if not queens.exists and not is_training_queen:
                # If there's no queen and not currently training one, check if we can afford one and have enough supply
                if self.can_afford(UnitID.QUEEN) and self.supply_left > 2:
                    # If we can, train a queen
                    self.do(th.train(UnitID.QUEEN))


    async def build_next_base(self):
        if self.minerals > 500:
            target = await self.get_next_expansion()
            if self.tag_worker_build_2nd_base == 0:
                if worker := self.mediator.select_worker(target_position=target):                
                    self.mediator.assign_role(tag=worker.tag, role=UnitRole.BUILDING)
                    self.tag_worker_build_2nd_base = worker
                    #self.mediator.build_with_specific_worker(worker, UnitID.HATCHERY, target, BuildingPurpose.NORMAL_BUILDING)
                    self.mediator.build_with_specific_worker(worker=self.tag_worker_build_2nd_base, structure_type=UnitID.HATCHERY, pos=target, building_purpose=BuildingPurpose.NORMAL_BUILDING)


    async def inject_larva(self):
        for queen in self.units(UnitID.QUEEN).ready:
            abilities = await self.get_available_abilities(queen)
            if AbilityId.EFFECT_INJECTLARVA in abilities and queen.energy >= 25:
                hatchery = self.townhalls.closest_to(queen.position)
                if queen.distance_to(hatchery) < 10:
                    self.do(queen(AbilityId.EFFECT_INJECTLARVA, hatchery))


#_______________________________________________________________________________________________________________________
#          DEBUG TOOL
#_______________________________________________________________________________________________________________________

    async def debug_tool(self):
        current_time = time.time()
        if current_time - self.last_debug_time >= 1:  # Se passou mais de um segundo
            print(self.mediator.get_all_enemy)
            print("Enemy Race: ", self.EnemyRace)
            print("Second Base: ", self.second_base)
            print("Guess Strategy: ", self.guess_strategy)
            #print("RallyPointSet: ", self.rally_point_set)
            #print("FirstBase: ", self.first_base)
            #print("SecondBase: ", self.second_base)
            self.last_debug_time = current_time  # Atualizar a última vez que a ferramenta de debug foi chamada

#_______________________________________________________________________________________________________________________
#          ON UNIT CREATED
#_______________________________________________________________________________________________________________________


    async def on_unit_created(self, unit: Unit) -> None:
        """
        Can use burnysc2 hooks as usual, just add a call to the
        parent method before your own logic.
        """
        await super(MyBot, self).on_unit_created(unit)


        # assign our forces ATTACKING by default
        if unit.type_id not in WORKER_TYPES and unit.type_id not in {
            UnitID.QUEEN,
            UnitID.MULE,
            UnitID.OVERLORD,
        }:
            # here we are making a request to an ares manager via the mediator
            # See https://aressc2.github.io/ares-sc2/api_reference/manager_mediator.html
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

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
        # MINE
        # ares-sc2 Mining behavior
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.mining.Mining
        self.register_behavior(Mining())

        # MAKE SUPPLY
        # ares-sc2 AutoSupply
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.auto_supply.AutoSupply
        if self.build_order_runner.build_completed:
            self.register_behavior(AutoSupply(base_location=self.start_location))

        # BUILD ARMY
        # ares-sc2 SpawnController

        if self.EnemyRace == Race.Terran:
            self.register_behavior(SpawnController(ARMY_COMP_VS_TERRAN[self.race]))

        else:
            self.register_behavior(SpawnController(ARMY_COMP[self.race]))


        # see also `ProductionController` for ongoing generic production, not needed here
        # https://aressc2.github.io/ares-sc2/api_reference/behaviors/macro_behaviors.html#ares.behaviors.macro.spawn_controller.ProductionController

        self._zerg_specific_macro()

#_______________________________________________________________________________________________________________________
#          DEF MICRO
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

            if self.race == Race.Zerg:
                # you can add a CombatManeuver to another CombatManeuver!!!
                burrow_behavior: CombatManeuver = self.burrow_behavior(unit)
                attacking_maneuver.add(burrow_behavior)

            # enemy around, engagement control
            if all_close:
                # ares's cython version of `cy_in_attack_range` is approximately 4
                # times speedup vs burnysc2's `all_close.in_attack_range_of`

                # idea here is to attack anything in range if weapon is ready
                # check for enemy units first
                if unit.type_id == UnitID.ROACH:
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

                else:
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
            burrow_maneuver.add(UseAbility(AbilityId.BURROWUP, roach, None))
        elif (
            not roach.is_burrowed
            and roach.health_percentage <= self.BURROW_AT_HEALTH_PERC
        ):
            burrow_maneuver.add(UseAbility(AbilityId.BURROWDOWN, roach, None))

        return burrow_maneuver



#_______________________________________________________________________________________________________________________
#          ZERG MACRO
#_______________________________________________________________________________________________________________________


    def _zerg_specific_macro(self) -> None:
        if self.EnemyRace == Race.Terran:
            if (not self.already_pending_upgrade(UpgradeId.ZERGLINGMOVEMENTSPEED)):
                self.research(UpgradeId.ZERGLINGMOVEMENTSPEED)
        else:        
            if (
                not self.already_pending_upgrade(UpgradeId.BURROW)
                and self.townhalls.idle
                and self.build_order_runner.build_completed
                and self.can_afford(UpgradeId.BURROW)
            ):
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