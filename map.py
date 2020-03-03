# Self Driving Car and maps with canvas creation
# Interface between Ai and graphics for self driving car 
import numpy as np

#importing required kivy libraries
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

#variabes for kerping the track of last position of car 
last_pos_x = 0
last_pos_y = 0


# Initializing the map
initial_update = True
def init():
    
    #declaration of variables for goal destination
    global sand
    global goal_pos_x
    global goal_pos_y
    global initial_update
    
    #Initializing the variables for goal destination
    sand = np.zeros((width,length))
    goal_pos_x = 20
    goal_pos_y = length - 20
    initial_update = False
    
    
# Current distance between the car and the goal destination (0 by default)
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
    
    # Detecting obstacles in front of the car
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    
    # Detecting obstacles on the left of the car
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    
    # Detecting obstacles on the right of the car
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        # Updates the current position using the velocity vector
        self.pos = Vector(*self.velocity) + self.pos
        
        # Rotates the current angle of the car using the rotation of the action
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        # Updates the position and angle of the sensors (30 is the distance between the car and the sensors)
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        
       
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.

# Graphical balls that represent the sensors on the map
class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass

# Creating the game class for visualization of the car in the app

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    
# function for the initialization of car app
    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(1, 0)
        
# updating the position of the car according to the signals
    def update(self, dt):

        global last_distance
        global goal_pos_x
        global goal_pos_y
        global width
        global length

        width = self.width
        length = self.height
        if initial_update:
            init()

        self.car.move(5)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

#Main class to initialize of whole simulation and App

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0)
        return parent


if __name__ == '__main__':
    CarApp().run()
