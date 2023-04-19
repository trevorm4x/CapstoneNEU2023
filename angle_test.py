from djitellopy import Tello

import time


if __name__ == '__main__':

    flight_time = float(input("Enter a flight time: \n"))
    speed = int(input("Enter a flight speed: \n"))

    drone = Tello()
    drone.connect()
    drone.takeoff()

    movescale = [0, 0, 0, 0]
    movescale[1] = speed
    drone.send_rc_control(*movescale)
    time.sleep(flight_time)
    drone.send_rc_control(0, 0, 0, 0)
    drone.land()
    drone.end()
