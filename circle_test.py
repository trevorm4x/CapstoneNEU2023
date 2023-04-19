from djitellopy import Tello
from time import sleep


N_ROUNDS = 1
DISTANCE = 200

tello = Tello()

if __name__ == '__main__':
    tello.connect()
    tello.takeoff()

    v = 0
    for _ in range(4):
        tello.send_rc_control(20, 0, v, -50)
        sleep(2)
        tello.send_rc_control(0, 0, 0, 0)
        sleep(0.5)

#    tello.curve_xyz_speed(
#        x1=-int(DISTANCE/2),
#        y1=int(DISTANCE/2),
#        z1=0,
#        x2=-DISTANCE,
#        y2=0,
#        z2=0,
#        speed=20
    # )

    tello.land()
    tello.end()
