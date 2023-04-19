from djitellopy import Tello


N_ROUNDS = 2
DISTANCE = 100

tello = Tello()

if __name__ == '__main__':
    tello.connect()
    tello.takeoff()

    for _ in range(N_ROUNDS):
        tello.move_forward(DISTANCE)
        tello.move_left(DISTANCE)
        tello.move_back(DISTANCE)
        tello.move_right(DISTANCE)

        for _ in range(4):
            tello.move_forward(DISTANCE)
            tello.rotate_counter_clockwise(90)

    tello.land()
