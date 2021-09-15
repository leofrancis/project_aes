#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    CTRL + W     : toggle constant velocity mode at 60 km/h

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change sensor position
    ` or N       : next sensor
    [1-9]        : change to sensor [1-9]
    G            : toggle radar visualization
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    V            : Select next map layer (Shift+V reverse)
    B            : Load current selected map layer (Shift+B to unload)

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

#send_iot_data = 0 # 0: dont send | 1: send data IoT

import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

import cantools
import can
import requests
import json
import threading
import time

from dotenv import load_dotenv, find_dotenv


try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_j
    from pygame.locals import K_k
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")
    name = lambda x: " ".join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + u"\u2026") if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args):
        load_dotenv(find_dotenv())
        self.world = carla_world
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("  The server could not send the OpenDRIVE (.xodr) file:")
            print(
                "  Make sure it exists, has the same name of your town, and is correct."
            )
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All,
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )
        # Get a random blueprint.
        blueprint = random.choice(
            self.world.get_blueprint_library().filter(self._actor_filter)
        )
        blueprint.set_attribute("role_name", self.actor_role_name)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color", color)
        if blueprint.has_attribute("driver_id"):
            driver_id = random.choice(
                blueprint.get_attribute("driver_id").recommended_values
            )
            blueprint.set_attribute("driver_id", driver_id)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "true")
        # set the max speed
        if blueprint.has_attribute("speed"):
            self.player_max_speed = float(
                blueprint.get_attribute("speed").recommended_values[1]
            )
            self.player_max_speed_fast = float(
                blueprint.get_attribute("speed").recommended_values[2]
            )
        else:
            print("No recommended values for 'speed' attribute")
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are no spawn points available in your map/town.")
                print("Please add some Vehicle Spawn Point to your UE4 scene.")
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = (
                random.choice(spawn_points) if spawn_points else carla.Transform()
            )
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification("LayerMap selected: %s" % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification("Unloading map layer: %s" % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification("Loading map layer: %s" % selected)
            self.world.load_map_layer(selected)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def modify_vehicle_physics(self, vehicle):
        physics_control = vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        vehicle.apply_physics_control(physics_control)

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        if self.radar_sensor is not None:
            self.toggle_radar()
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
        ]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (
                    event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT
                ):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_j:
                    world.hud.send_iot_data = 1
                    #CREATING SEARINGS
                    print("CREATING SERIES...")
                    req = REQUEST()
                    var = get_var_request()

                    t0_stored = int(get_env('T0'))
                    t1_stored = int(get_env('T1'))
                    t_now = int(time.time())

                    if (t_now * (10 ** 6) >= t0_stored and t_now * (10 ** 6) <= t1_stored):
                        t_last = ((t1_stored / (10 ** 6)) - t_now )/60

                        # if (t_last < 2):
                        #     print("SERIES UPDATED!")
                        #     t_end = t_now + (60*20)
                        #
                        #     #update_env("T0", str(t0_stored))
                        #     update_env("T1", str(t_end * (10 ** 6)))
                        #
                        #     for i in var:
                        #         qty = i["qty"]
                        #         unit = i["unit"]
                        #
                        #         for dev in range(0, qty):
                        #             time.sleep(0.5)
                        #             req.create(unit, dev)
                        # else:
                        print(f"SERIES ALREADY CREATED [Still {t_last} min]!")

                    else:
                        print("SERIES CREATED!")
                        t_end = t_now + (60*20)

                        update_env("T0", str(t_now * (10 ** 6)))
                        update_env("T1", str(t_end * (10 ** 6)))

                        for i in var:
                            qty = i["qty"]
                            unit = i["unit"]

                            for dev in range(0, qty):
                                time.sleep(0.5)
                                req.create(unit, dev)

                elif event.key == K_k:
                    world.hud.send_iot_data = 0
                    world.hud.can_thread.stop_thread()
                    print("FINISHING SERIES...")
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification(
                            "Enabled Constant Velocity Mode at 60 km/h"
                        )
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.recording_enabled:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file(
                        "manual_recording.rec", world.recording_start, 0, 0
                    )
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start)
                    )
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification(
                        "Recording start time is %d" % (world.recording_start)
                    )
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = (
                            not self._control.manual_gear_shift
                        )
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification(
                            "%s Transmission"
                            % (
                                "Manual"
                                if self._control.manual_gear_shift
                                else "Automatic"
                            )
                        )
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            "Autopilot %s"
                            % ("On" if self._autopilot_enabled else "Off")
                        )
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else:  # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else:  # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if (
                    current_lights != self._lights
                ):  # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(
                    pygame.key.get_pressed(), clock.get_time(), world
                )
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = 0.01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = 0.01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = (
                world.player_max_speed_fast
                if pygame.key.get_mods() & KMOD_SHIFT
                else world.player_max_speed
            )
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- REQUEST--------------------------------------------------------------------
# ==============================================================================
class REQUEST(object):
    def __init__(self):
        print('Request [Started]')
        self.CLIENT_CERTIFICATE = ['./certificate/client-42-A7B64D415BD3E9A6.pem',
                                   './certificate/client-42-A7B64D415BD3E9A6.key']

        self.session = requests.Session()
        self.session.headers = {'Content-type' : 'application/json'}
        self.session.cert = self.CLIENT_CERTIFICATE

    def send(self, currentTime, value, unit, dev):
        #print('Request [SEND]')
        time.sleep(0.5)

        self.put(unit, dev, currentTime, value)

    def get(self, unit, dev):
        method = 'GET'
        url = self.url(method)

        print(f'Request [{method}]')

        json_query = self.query(method, unit, dev, None, None)

        response = self.session.post(url, json.dumps(json_query))

        valid = response.ok
        data = response.json()

        if (valid and len(data["series"]) > 0):
            return True
        else:
            return False


    def create(self, unit, dev):
        method = 'CREATE'
        url = self.url(method)

        print(f'Request [{method}]')

        json_query = self.query(method, unit, dev, None, None)

        print(json_query) #to get the t0 e t1

        response = self.session.post(url, json.dumps(json_query))

        valid = response.ok
        status_code = response.status_code

        if (valid and status_code == 400):
            return True
        else:
            return False


    def put(self, unit, dev, currentTime, value):
        method = 'PUT'
        url = self.url(method)

        time_start = int(time.time())


        t1 = int(get_env('T1')) / (10 ** 6)

        if t1 < time_start:
            print("RESTART SERIES...")

        print(f'Request [{method}]')

        json_query = self.query(method, unit, dev, currentTime, value)

        #print(json_query)

        response = self.session.post(url, json.dumps(json_query))

        valid = response.ok
        status_code = response.status_code

        time_end = int(time.time())
        #print(response)
        request_time = time_end - time_start
        if (valid and status_code == 204):
            print(f"{method}-True | Time-{request_time}s")
            return True
        else:
            print(f"{method}-False | Time-{request_time}s")
            return False

    def url(self, method):
        method = method.lower()

        result =f"https://iot.lisha.ufsc.br/api/{method}.php"

        return result

    def query(self, method, unit, dev, currentTime, value):
        version = 1.1
        # unit = 0x84963924 # 2147473648

        #t0 = 1627127115000000  # Sunday, July 24, 2021 8:45:15 AM GMT-03:00
        #t1 = 1658663115000000  # Sunday, July 24, 2022 8:45:15 AM GMT-03:00
        r = 0
        x = 10
        y = 10
        z = 10
        #dev = 0
        error = 0
        trust = 0
        wf = 0
        domain =  get_env('DOMAIN')
        username = get_env('USERNAME')
        password = get_env('PASSWORD')
        t0 = int(get_env('T0'))
        t1 = int(get_env('T1'))

        # if (method == "CREATE"):
            # t_base = int(time.time())
            # t0 = t_base * (10 ** 6)
            # t1 = (t_base + (60*5)) * (10 ** 6) # 5 minutos

        if (method == "GET"):
            json = {
                "series": {
                    "version": version,
                    "t0": t0,
                    "t1": t1,
                    "unit": unit,
                    "r": r,
                    "x": x,
                    "y": y,
                    "z": z,
                    "dev": dev,
                    "workflow": wf,
                },
                "credentials": {"domain": domain},
            }
        elif (method == "CREATE"):
            json = {
                "series": {
                    "version": version,
                    "t0": t0,
                    "t1": t1,
                    "unit": unit,
                    "r": r,
                    "x": x,
                    "y": y,
                    "z": z,
                    "dev": dev,
                    "workflow": wf
                },
                "credentials": {"domain": domain},
            }
        elif (method == "PUT"):
            json = {
                "smartdata": [
                    {
                        "version": version,
                        "t": currentTime,
                        "value": value,
                        "unit": unit,
                        "error": error,
                        "confidence": trust,
                        "r": r,
                        "x": x,
                        "y": y,
                        "z": z,
                        "dev": dev
                    }]
                # ],
                # "credentials": {
                #     "domain": domain,
                # },
            }
        elif (method == "SEARCH"):
            json = {
                "series": {
                    "version": version,
                    "t0": t0,
                    "t1": t1,
                    "unit": unit,
                    "r": r,
                    "x": x,
                    "y": y,
                    "z": z,
                    "dev": dev,
                    "workflow": wf,
                },
                "credentials": {
                    "domain": domain,
                    "username": username,
                    "password": password,
                },
            }

        return json

# ==============================================================================
# -- UTILITY FUNCTIONS ----------------------------------------------------------------
# ==============================================================================
def get_env(key):
    with open(".env", "r") as f:
        for line in f.readlines():
            try:
                key_line, value_line = line.split('=')
                if key == key_line:
                    #print(value_line)
                    value_line = value_line.replace("\n", "")
                    value_line = value_line.replace('"', "")
                    return value_line
            except ValueError:
                pass

def update_env(key, new_value):
    result = ""
    with open(".env", "r") as f:
        for line in f.readlines():
            try:
                #print(line)
                key_line, value_line = line.split('=')
                if key == key_line:
                    result = result + key + '="' + new_value +'"' + "\n"
                    #print(result)
                else:
                    result = result + line
            except ValueError:
                pass

    with open(".env", "w") as f:
        f.write(result)

def get_var_request():
    var = [{"id": 464,  "name": "speed",         "unit": 0xE4963924, "SI Unit": "m/s",          "qty": 1},
           {"id": 465,  "name": "gyroscope",     "unit": 0xE4B23924, "SI Unit": "rad/s",        "qty": 3},
           {"id": 466,  "name": "accelerometer", "unit": 0xE4962924, "SI Unit": "m/s2",         "qty": 3},
           {"id": 467,  "name": "gnss-altitude", "unit": 0xE4964924, "SI Unit": "m",            "qty": 1},
           {"id": 467,  "name": "gnss-latitude", "unit": 0xE4B24924, "SI Unit": "degress->RAD", "qty": 1},
           {"id": 467,  "name": "gnss-longitude","unit": 0xE4B24924, "SI Unit": "degress->RAD", "qty": 1},
           {"id": 468,  "name": "gear",          "unit": 0xF8000006, "SI Unit": "Counter",      "qty": 1},
           {"id": 469,  "name": "colision",      "unit": 0xF8000006, "SI Unit": "Counter",      "qty": 1}]

    return var

def get_unit_request(id, name):
    var = get_var_request()

    for i in var:
        if (id != 467):
            if (i["id"]==id):
                return i["unit"]
        else:
            if (i["name"]==name):
                return i["unit"]

# ==============================================================================
# -- CAN THREAD ----------------------------------------------------------------
# ==============================================================================
def update(running, send_iot_data):
    print('CAN Thread[Started]')

    db = cantools.database.load_file("./_honda_2017.dbc")
    can_bus = can.interface.Bus("vcan0", bustype="socketcan")

    req = REQUEST()

    speed = 0
    speed_ant = 0

    gear = 0
    gear_ant = 0

    gyro = ''
    gyro_ant = ''

    accel = ''
    accel_ant = ''

    gnss = ''
    gnss_ant = ''

    vehicle_col = 0
    vehicle_col_ant = 0

    #IDS
    speed_id = 464
    gyroscope_id = 465
    accelerometer_id = 466
    gnss_id = 467
    gear_id = 468
    colision_id = 469
    #print("run - ",running)

    while True:
        #print("run")
        if (running  == True):
            break

        if (send_iot_data == 1):

            message = can_bus.recv()

            # SPEED
            if (message.arbitration_id == speed_id):
                speed = float(db.decode_message(message.arbitration_id, message.data)['WHEEL_SPEED'])
                currentTime = int(time.time()) * (10 ** 6)

                if (speed_ant != speed):
                    speed_ant = speed
                    unit = get_unit_request(speed_id, 'speed')
                    # print('Speed[currentTime]: ', currentTime)
                    # print('Speed[value]: ', speed)
                    req.send(currentTime, speed, unit, 0)

            # GYROSCOPE
            if (message.arbitration_id == gyroscope_id):
                gyro_x = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_X'])
                gyro_y = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_Y'])
                gyro_z = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_Z'])
                gyro_x_signal = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_X_SIG'])
                gyro_y_signal = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_Y_SIG'])
                gyro_z_signal = float(db.decode_message(message.arbitration_id, message.data)['GYROSCOPE_Z_SIG'])

                if gyro_x_signal > 0:
                    gyro_x = gyro_x * -1

                if gyro_y_signal > 0:
                    gyro_y = gyro_y * -1

                if gyro_z_signal > 0:
                    gyro_z = gyro_z * -1

                gyro = str(gyro_x) + '|' + str(gyro_y) + '|' + str(gyro_z)
                currentTime = int(time.time()) * (10 ** 6)

                if (gyro != gyro_ant):
                    gyro_ant = gyro
                    # print('GYROSCOPE[currentTime]: ', currentTime)
                    # print('GYROSCOPE[value]: ', gyro)

                    unit = get_unit_request(gyroscope_id, 'gyroscope')

                    req.send(currentTime, gyro_x, unit, 0)
                    req.send(currentTime, gyro_y, unit, 1)
                    req.send(currentTime, gyro_z, unit, 2)

            # ACCELERO
            if (message.arbitration_id == accelerometer_id):
                accel_x = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_X'])
                accel_y = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_Y'])
                accel_z = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_Z'])
                accel_x_signal = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_X_SIG'])
                accel_y_signal = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_Y_SIG'])
                accel_z_signal = float(db.decode_message(message.arbitration_id, message.data)['ACCELERO_Z_SIG'])

                if accel_x_signal > 0:
                    accel_x = accel_x * -1

                if accel_y_signal > 0:
                    accel_y = accel_y * -1

                if accel_z_signal > 0:
                    accel_z = accel_z * -1

                accel = str(accel_x) + '|' + str(accel_y) + '|' + str(accel_z)
                currentTime = int(time.time()) * (10 ** 6)

                if (accel != accel_ant):
                    accel_ant = accel
                    # print('ACCELERO[currentTime]: ', currentTime)
                    # print('ACCELERO[value]: ', accel)

                    unit = get_unit_request(accelerometer_id, 'accelerometer')

                    req.send(currentTime, accel_x, unit, 0)
                    req.send(currentTime, accel_y, unit, 1)
                    req.send(currentTime, accel_z, unit, 2)

            # GNSS
            if (message.arbitration_id == gnss_id):
                gnss_alt = float(db.decode_message(message.arbitration_id, message.data)['GNSS_ALTITUDE'])
                gnss_lat = float(db.decode_message(message.arbitration_id, message.data)['GNSS_LATITUDE']) / 1000
                gnss_lon = float(db.decode_message(message.arbitration_id, message.data)['GNSS_LONGITUDE']) / 1000
                gnss_lat_signal = float(db.decode_message(message.arbitration_id, message.data)['GNSS_LATITUDE_SIG'])
                gnss_lon_signal = float(db.decode_message(message.arbitration_id, message.data)['GNSS_LONGITUDE_SIG'])


                if gnss_lat_signal > 0:
                    gnss_lat = gnss_lat * -1

                if gnss_lon_signal > 0:
                    gnss_lon = gnss_lon * -1

                rad = 0.0174533

                #Convert o Radians (rad)
                gnss_lat = gnss_lat * rad
                gnss_lon = gnss_lon * rad

                gnss = str(gnss_alt) + '|' + str(gnss_lat) + '|' + str(gnss_lon)
                currentTime = int(time.time()) * (10 ** 6)

                if (gnss != gnss_ant):
                    gnss_ant = gnss
                    # print('GNSS[currentTime]: ', currentTime)
                    # print('GNSS[value]: ', gnss)

                    unit_alt = get_unit_request(gnss_id, 'gnss-altitude')
                    unit_lat = get_unit_request(gnss_id, 'gnss-latitude')
                    unit_lon = get_unit_request(gnss_id, 'gnss-longitude')

                    req.send(currentTime, gnss_lat, unit_lat, 0)
                    req.send(currentTime, gnss_lon, unit_lon, 0)
                    req.send(currentTime, gnss_alt, unit_alt, 0)

            # GEAR
            if (message.arbitration_id == gear_id):
                gear = float(db.decode_message(message.arbitration_id, message.data)['GEAR']) - 1
                currentTime = int(time.time()) * (10 ** 6)

                if (gear_ant != gear):
                    gear_ant = gear
                    # print('GEAR[currentTime]: ', currentTime)
                    # print('GEAR[value]: ', gear)

                    unit = get_unit_request(gear_id, 'gear')

                    req.send(currentTime, gear, unit, 0)

            # VEHICLE COLISION
            if (message.arbitration_id == colision_id):
                vehicle_col = float(db.decode_message(message.arbitration_id, message.data)['COLISION'])
                currentTime = int(time.time()) * (10 ** 6)

                if (vehicle_col_ant != vehicle_col):
                    vehicle_col_ant = vehicle_col
                    # print('VEHICLE_COL[currentTime]: ', currentTime)
                    # print('VEHICLE_COL[value]: ', vehicle_col)

                    unit = get_unit_request(colision_id, 'colision')

                    req.send(currentTime, vehicle_col, unit, 1)


class CAN_THREAD(object):
    def __init__(self, send_iot_data):
        self.running = False
        self.thread = threading.Thread(name='update', target=update, args=(lambda: self.running, send_iot_data))
        self.thread.start()

    def stop_thread(self):
        self.running = True
        self.thread.join()
# ==============================================================================
# -- CAN (INSERT) --------------------------------------------------------------
# ==============================================================================


class CAN(object):
    def __init__(self):
        #self.db = cantools.database.load_file("/home/leonardo/Documents/GitHub/opendbc/generator/honda/honda_clarity_hybrid_2018_can.dbc")
        self.db = cantools.database.load_file("./_honda_2017.dbc")
        self.can_bus = can.interface.Bus("vcan0", bustype="socketcan")

        self.speed_message = self.db.get_message_by_name('WHEEL_SPEEDS')
        self.gear_message = self.db.get_message_by_name('GEAR_POS')
        self.gyroscope_message = self.db.get_message_by_name('GYROSCOPE')
        self.accelerometer_message = self.db.get_message_by_name('ACCELERO')
        self.gnss_message = self.db.get_message_by_name('GNSS')
        self.vehicle_colision_message = self.db.get_message_by_name('VEHICLE_COLISION')

        print("CAN [Started]")

    def send_speed(self, speed):
        data = self.speed_message.encode({'WHEEL_SPEED': speed})
        message = can.Message(arbitration_id=self.speed_message.frame_id, data=data)
        self.can_bus.send(message)

    def send_gear(self, gear):
        data = self.gear_message.encode({'GEAR': gear})
        message = can.Message(arbitration_id=self.gear_message.frame_id, data=data)
        self.can_bus.send(message)

    def send_gyro(self, gyro_x, gyro_x_sig, gyro_y, gyro_y_sig, gyro_z, gyro_z_sig):
        data = self.gyroscope_message.encode({'GYROSCOPE_X': float(gyro_x), 'GYROSCOPE_Y': float(gyro_y), 'GYROSCOPE_Z': float(gyro_z), 'GYROSCOPE_X_SIG': gyro_x_sig, 'GYROSCOPE_Y_SIG': gyro_y_sig, 'GYROSCOPE_Z_SIG': gyro_z_sig})
        message = can.Message(arbitration_id=self.gyroscope_message.frame_id, data=data)
        self.can_bus.send(message)

    def send_accel(self, accel_x, accel_x_sig, accel_y, accel_y_sig, accel_z, accel_z_sig):
        # print("send_accel: "+str(accel_x_sig)+ '_'+str(accel_x)+"|"+str(accel_y_sig)+ '_'+str(accel_y)+"|"+str(accel_z_sig)+"_"+str(accel_z))

        data = self.accelerometer_message.encode({'ACCELERO_X': float(accel_x), 'ACCELERO_Y': float(accel_y), 'ACCELERO_Z': float(accel_z), 'ACCELERO_X_SIG': accel_x_sig, 'ACCELERO_Y_SIG': accel_y_sig, 'ACCELERO_Z_SIG': accel_z_sig})
        message = can.Message(arbitration_id=self.accelerometer_message.frame_id, data=data)
        self.can_bus.send(message)

    def send_gnss(self, gnss_altitude, gnss_latitude, gnss_latitude_sig, gnss_longitude, gnss_longitude_sig):
        #print("GNSS_SEND: "+str(gnss_altitude)+"|"+str(gnss_latitude_sig)+ '_'+str(gnss_latitude)+"|"+str(gnss_longitude_sig)+"_"+str(gnss_longitude))


        data = self.gnss_message.encode({'GNSS_ALTITUDE': float(gnss_altitude), 'GNSS_LATITUDE': float(gnss_latitude), 'GNSS_LONGITUDE': float(gnss_longitude), 'GNSS_LATITUDE_SIG': gnss_latitude_sig, 'GNSS_LONGITUDE_SIG': gnss_longitude_sig})

        message = can.Message(arbitration_id=self.gnss_message.frame_id, data=data)
        self.can_bus.send(message)

    def send_colision(self, vehicle_colision):
        data = self.vehicle_colision_message.encode({'COLISION': vehicle_colision})
        message = can.Message(arbitration_id=self.vehicle_colision_message.frame_id, data=data)
        self.can_bus.send(message)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = "courier" if os.name == "nt" else "mono"
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == "nt" else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.create_can = True
        self.send_iot_data = 0

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = "N" if compass > 270.5 or compass < 89.5 else ""
        heading += "S" if 90.5 < compass < 269.5 else ""
        heading += "E" if 0.5 < compass < 179.5 else ""
        heading += "W" if 180.5 < compass < 359.5 else ""
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter("vehicle.*")

        if (self.create_can == True and self.send_iot_data == 1):
            #print("created")
            self.can = CAN()
            self.can_thread = CAN_THREAD(self.send_iot_data)
            self.create_can = False

        if (self.send_iot_data == 1):
            # CAN Bus
        	speed = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        	speed = round(speed, 2)
        	self.can.send_speed(speed)
        	self.can.send_gear(c.gear + 1)

        	# print("Gyro: ",world.imu_sensor.gyroscope)

        	gyro_x = round(world.imu_sensor.gyroscope[0], 2)
        	gyro_y = round(world.imu_sensor.gyroscope[1], 2)
        	gyro_z = round(world.imu_sensor.gyroscope[2], 2)

        	if (gyro_x >= 0):
        	   gyro_x_sig = 0
        	else:
        	   gyro_x_sig = 1

        	if (gyro_y >= 0):
        	   gyro_y_sig = 0
        	else:
        	   gyro_y_sig = 1

        	if (gyro_z >= 0):
        	   gyro_z_sig = 0
        	else:
        	   gyro_z_sig = 1

        	self.can.send_gyro(abs(gyro_x), gyro_x_sig, abs(gyro_y), gyro_y_sig, abs(gyro_z), gyro_z_sig)

        	accel_x = float(round(world.imu_sensor.accelerometer[0], 2))
        	accel_y = float(round(world.imu_sensor.accelerometer[1], 2))
        	accel_z = float(round(world.imu_sensor.accelerometer[2], 2))

        	# Setting 0 if it's NULL
        	if (accel_x == None or accel_x > 9999):
        	    accel_x = 0

        	if (accel_y == None or accel_y > 9999):
        	    accel_y = 0

        	if (accel_z == None or accel_z > 9999):
        	    accel_z = 0

        	# Getting Signal for each field
        	if (accel_x >= 0):
        	   accel_x_sig = 0
        	else:
        	   accel_x_sig = 1

        	if (accel_y >= 0):
        	   accel_y_sig = 0
        	else:
        	   accel_y_sig = 1

        	if (accel_z >= 0):
        	   accel_z_sig = 0
        	else:
        	   accel_z_sig = 1

        	self.can.send_accel(abs(accel_x), accel_x_sig, abs(accel_y), accel_y_sig, abs(accel_z), accel_z_sig)

        	gnss_alt = float(round(t.location.z, 6))
        	gnss_lat = float(round(world.gnss_sensor.lat, 6) * 1000)
        	gnss_lon = float(round(world.gnss_sensor.lon, 6) * 1000)

        	gnss_lat_sig = 0
        	gnss_lon_sig = 0

        	if (gnss_alt < 0):
        	    gnss_alt = 0
        	elif (gnss_alt > 9999):
        	    gnss_alt = 9999

        	if (gnss_lat > 9999):
        	    gnss_lat = 9999

        	if (gnss_lon > 9999):
        	    gnss_lon = 9999

        	if (gnss_lat >= 0):
        	    gnss_lat_sig = 0
        	else:
        	    gnss_lat_sig = 1

        	if (gnss_lon >= 0):
        	    gnss_lon_sig = 0
        	else:
        	    gnss_lon_sig = 1

        	#print("GNSS: "+str(gnss_alt)+"|"+str(gnss_lat_sig)+ '_'+str(gnss_lat)+"|"+str(gnss_lon_sig)+"_"+str(gnss_lon))

        	self.can.send_gnss(gnss_alt, abs(gnss_lat), gnss_lat_sig, abs(gnss_lon), gnss_lon_sig)

        self._info_text = [
            "Server:  % 16.0f FPS" % self.server_fps,
            "Client:  % 16.0f FPS" % clock.get_fps(),
            "",
            "Vehicle: % 20s" % get_actor_display_name(world.player, truncate=20),
            "Map:     % 20s" % world.map.name,
            "Simulation time: % 12s"
            % datetime.timedelta(seconds=int(self.simulation_time)),
            "",
            "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u"Compass:% 17.0f\N{DEGREE SIGN} % 2s" % (compass, heading),
            "Accelero: (%5.1f,%5.1f,%5.1f)" % (world.imu_sensor.accelerometer),
            "Gyroscop: (%5.1f,%5.1f,%5.1f)" % (world.imu_sensor.gyroscope),
            "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
            "GNSS:% 24s"
            % ("(% 2.6f, % 3.6f)" % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            "Height:  % 18.0f m" % t.location.z,
            "",
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ("Throttle:", c.throttle, 0.0, 1.0),
                ("Steer:", c.steer, -1.0, 1.0),
                ("Brake:", c.brake, 0.0, 1.0),
                ("Reverse:", c.reverse),
                ("Hand brake:", c.hand_brake),
                ("Manual:", c.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear),
            ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [("Speed:", c.speed, 0.0, 5.556), ("Jump:", c.jump)]
        self._info_text += [
            "",
            "Collision:",
            collision,
            "",
            "Number of vehicles: % 8d" % len(vehicles),
        ]
        if len(vehicles) > 1:
            self._info_text += ["Nearby vehicles:"]
            distance = lambda l: math.sqrt(
                (l.x - t.location.x) ** 2
                + (l.y - t.location.y) ** 2
                + (l.z - t.location.z) ** 2
            )
            vehicles = [
                (distance(x.get_location()), x)
                for x in vehicles
                if x.id != world.player.id
            ]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append("% 4dm %s" % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text("Error: %s" % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [
                            (x + 8, v_offset + 8 + (1.0 - y) * 30)
                            for x, y in enumerate(item)
                        ]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(
                            display, (255, 255, 255), rect, 0 if item[1] else 1
                        )
                    else:
                        rect_border = pygame.Rect(
                            (bar_h_offset, v_offset + 8), (bar_width, 6)
                        )
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8),
                                (6, 6),
                            )
                        else:
                            rect = pygame.Rect(
                                (bar_h_offset, v_offset + 8), (f * bar_width, 6)
                            )
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: CollisionSensor._on_collision(weak_self, event)
        )

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification("Collision with %r" % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)

        self.hud.can.send_colision(intensity)

        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda event: LaneInvasionSensor._on_invasion(weak_self, event)
        )

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.spawn_actor(
            bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent
        )
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data)
        )

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        )
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))),
        )
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", str(35))
        bp.set_attribute("vertical_fov", str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=2.8, z=1.0), carla.Rotation(pitch=5)),
            attach_to=self._parent,
        )
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data)
        )

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll,
                ),
            ).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range  # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b),
            )


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        Attachment = carla.AttachmentType
        self._camera_transforms = [
            (
                carla.Transform(
                    carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)
                ),
                Attachment.SpringArm,
            ),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (
                carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                Attachment.SpringArm,
            ),
            (
                carla.Transform(
                    carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)
                ),
                Attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),
                Attachment.Rigid,
            ),
        ]
        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB", {}],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)", {}],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)", {}],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
                {},
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
                {},
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
                {},
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)", {"range": "50"}],
            ["sensor.camera.dvs", cc.Raw, "Dynamic Vision Sensor", {}],
            [
                "sensor.camera.rgb",
                cc.Raw,
                "Camera RGB Distorted",
                {
                    "lens_circle_multiplier": "3.0",
                    "lens_circle_falloff": "3.0",
                    "chromatic_aberration_intensity": "0.5",
                    "chromatic_aberration_offset": "0",
                },
            ],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                bp.set_attribute("image_size_x", str(hud.dim[0]))
                bp.set_attribute("image_size_y", str(hud.dim[1]))
                if bp.has_attribute("gamma"):
                    bp.set_attribute("gamma", str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith("sensor.lidar"):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == "range":
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors[index][2] != self.sensors[self.index][2])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        elif self.sensors[self.index][0].startswith("sensor.camera.dvs"):
            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(
                image.raw_data,
                dtype=np.dtype(
                    [
                        ("x", np.uint16),
                        ("y", np.uint16),
                        ("t", np.int64),
                        ("pol", np.bool),
                    ]
                ),
            )
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[
                dvs_events[:]["y"], dvs_events[:]["x"], dvs_events[:]["pol"] * 2
            ] = 255
            self.surface = pygame.surfarray.make_surface(dvs_img.swapaxes(0, 1))
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        display.fill((0, 0, 0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        print("finalizou")

        if world and world.recording_enabled:
            client.stop_recorder()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(description="CARLA Manual Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "-a", "--autopilot", action="store_true", help="enable autopilot"
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="1280x720",
        help="window resolution (default: 1280x720)",
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "--rolename",
        metavar="NAME",
        default="hero",
        help='actor role name (default: "hero")',
    )
    argparser.add_argument(
        "--gamma",
        default=2.2,
        type=float,
        help="Gamma correction of the camera (default: 2.2)",
    )
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")

if __name__ == "__main__":

    main()
