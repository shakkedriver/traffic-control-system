from Car import Car

ACCELERATION_FACTOR = 1
DECELERATION_FACTOR = -2

class NormalCar(Car):

    def update_speed(self, acceleration):
        if acceleration > 0:
            self.speed += ACCELERATION_FACTOR
        elif acceleration < 0:
            self.speed += DECELERATION_FACTOR
        self.speed = max(0, min(self.max_speed, self.speed))
