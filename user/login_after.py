# -*- coding: utf-8 -*-
from kivy.uix.screenmanager import Screen


class AfterLogin(Screen):
    pass

    @staticmethod
    def print_something():
        print('print something')

    def show_hide_pass(self):
        from user.login import Login
        Login.show_hide_pass(self)
