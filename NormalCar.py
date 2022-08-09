from Car import Car


class NormalCar(Car):


    def update_speed(self, acceleration):
        self.speed += acceleration
        self.speed = max(0, min(self.max_speed, self.speed))
