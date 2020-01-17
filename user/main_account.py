# -*- coding: utf-8 -*-
from kivy.app import App
from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.screenmanager import ScreenManager
from user.login import Login
from user.account import Account
from user.login_after import AfterLogin

error_message = StringProperty('')
username_after = ObjectProperty(None)
username = ObjectProperty(None)

Builder.load_file("login.kv")
Builder.load_file("account.kv")
Builder.load_file("after_login.kv")
Builder.load_file("themedwidgets.kv")

# Create the screen manager
sm = ScreenManager()
sm.add_widget(Login(name='login'))
sm.add_widget(Account(name='create_account'))
sm.add_widget(AfterLogin(name='after_login'))


class LoginApp(App):
    def build(self):
        return sm


if __name__ == "__main__":

    LoginApp().run()
