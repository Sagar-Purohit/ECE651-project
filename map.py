# Self Driving Car and maps with canvas creation
# Interface between Ai and graphics for self driving car
import numpy
from random import random, randint
import matplotlib.pyplot as plt
import time

#importing required kivy libraries
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

from Qlearning import DeepQNetwork

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


#variabes for kerping the track of last position of car
last_pos_x = 0
last_pos_y = 0
n_points = 0
length = 0


#initialisation of Deep Q network(brain)
brain = DeepQNetwork(5, 3, 0.9)

# angles for the rotation of our car for each step it changes
rotations_of_actions = [0, 20, -20]

# last reward that car had achieved
last_reward = 0

# initializing array of rewards
scores = []

# Initializing the map
initial_update = True

def init():
    #declaration and initiazation of variables for goal destination
    global goal_pos_x
    global goal_pos_y
    goal_pos_x = 20
    goal_pos_y = width - 20

    global initial_update
    initial_update = False

    global sand
    sand = numpy.zeros((length, width))


# Current distance between the car and the goal destination
last_distance = 0
# Creating the car class with sensors and positioning them
class Car(Widget):

    # Angle between x-axis and the direction of the car
    angle = NumericProperty(0)

    # Angle of the last rotation taken by the car (one of the angles in rotations_of_actions)
    rotation = NumericProperty(0)

    # Velocity coordinates and vector
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    # checking for obstacles on front
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)

    # checking for obstacles on front left
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)

    # checking for obstacles on front right
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)

    # Signals will be used as the reward of the brain giving 1 as bad and 0 as good
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # changes the current position of the car
        self.pos = Vector(*self.velocity) + self.pos

        # Rotates the car
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # updates sensor's position long with the car
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos

        # Caculates the new sand density around each sensor
        self.signal1 = int(numpy.sum(sand[int(self.sensor1_x) - 10:int(self.sensor1_x) + 10, int(self.sensor1_y) - 10:int(self.sensor1_y) + 10])) / 400.
        self.signal2 = int(numpy.sum(sand[int(self.sensor2_x) - 10:int(self.sensor2_x) + 10, int(self.sensor2_y) - 10:int(self.sensor2_y) + 10])) / 400.
        self.signal3 = int(numpy.sum(sand[int(self.sensor3_x) - 10:int(self.sensor3_x) + 10, int(self.sensor3_y) - 10:int(self.sensor3_y) + 10])) / 400.

        # checking if the car is near to the edge
        if self.sensor1_x > length - 10 or self.sensor1_x < 10 or self.sensor1_y > width - 10 or self.sensor1_y < 10:
            self.signal1 = 1.
        if self.sensor2_x > length - 10 or self.sensor2_x < 10 or self.sensor2_y > width - 10 or self.sensor2_y < 10:
            self.signal2 = 1.
        if self.sensor3_x > length - 10 or self.sensor3_x < 10 or self.sensor3_y > width - 10 or self.sensor3_y < 10:
            self.signal3 = 1.


# using kivy balls to be used as sensors in the ourr graphical represention
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass


class Game(Widget):

    car = ObjectProperty(None)

    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_pos_x
        global goal_pos_y
        global length
        global width

        length = self.width
        width = self.height

        if initial_update:
            init()

        xx = goal_pos_x - self.car.x
        yy = goal_pos_y - self.car.y

        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.

        #inputs to our brain Q deep learning algorithm
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]

        action = brain.update(last_reward, last_signal)

        # appending last reward to the array of score
        scores.append(brain.score())

        # deciding which rotation to be used from range input given
        rotation = rotations_of_actions[action]

        #changing car's movement upon deciding of the rotation angle
        self.car.move(rotation)

        # new distace from car to destination once updated the car's position
        distance = numpy.sqrt((self.car.x - goal_pos_x) ** 2 + (self.car.y - goal_pos_y) ** 2)

        # changing sensors along with car
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        # checking for the car if it has hit the edge of road and giving negetive reward for hitting
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else:
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)

            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        # setting new goal if car has arrived at destination
        if distance < 100:
            goal_pos_x = self.width - goal_pos_x
            goal_pos_y = self.height - goal_pos_y

        last_distance = distance


# Using kivy tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_pos_x, last_pos_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_pos_x = int(touch.x)
            last_pos_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_pos_x, last_pos_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += numpy.sqrt(max((x - last_pos_x) ** 2 + (y - last_pos_y) ** 2, 2))
            n_points += 1.
            density = n_points / (length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_pos_x = x
            last_pos_y = y


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0 / 30.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent
    
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = numpy.zeros((length,width))

    # Saves the AI
    def save(self, obj):
        print("Saving the AI...")
        brain.save()
        plt.plot(scores)
        plt.show()

    # Loads the AI
    def load(self, obj):
        print("Loading the last saved AI...")
        brain.load()

#Initialisation of code
if __name__ == '__main__':
    CarApp().run()
