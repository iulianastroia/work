# -*- coding: utf-8 -*-
import os
from kivy.app import App
from kivy.core.window import Window
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kniteditor.localization import _, change_language_to, current_language
from user import mysql_conn as db_connection, mysql_conn


class Login(Screen):
    pass

    @staticmethod
    def set_username():
        print("setting username")
        screens = App.get_running_app().root.screens
        other_screen = None
        db_user = ""
        for screen in screens:
            if screen.name == "login":
                db_user = screen.username.text
            elif screen.name == "after_login":
                other_screen = screen

        # copy existent values from db to edit account section
        other_screen.username_after.text = db_user
        other_screen.first_name.text, other_screen.last_name.text, other_screen.email.text, other_screen.phone.text, other_screen.password.text = mysql_conn.edit_user(
            db_user)

    def get_user(self):
        return self.username.text

    def store_username(self):
        store_current_user = open("user/current_user.txt", "w+")  # create file or write existent file
        username = self.username.text
        # store_current_user.write(self.username.text)
        store_current_user.write("Current username is: " + username)
        store_current_user.close()

    @staticmethod
    def read_username():
        if os.path.exists("user/current_user.txt"):
            read_file = open("user/current_user.txt", "r")  # read username from file
            for word in open("user/current_user.txt"):
                get_word = word.split(" ")
                save_username = get_word[-1]

            # get username from file(username is at the last position)
            print('saved username: ', save_username)
            read_file.close()
            return save_username

    @staticmethod
    def delete_username():  # TODO at LOGOUT
        if os.path.exists("user/current_user.txt"):
            os.remove("user/current_user.txt")  # delete file

    @staticmethod
    def translate():
        print(current_language())
        change_language_to("en")

    @staticmethod
    def translate_ro():
        print(current_language())
        change_language_to("ro")

    def show_hide_pass(self):
        print("Login show_hide_pass")
        if self.show_password.active:
            print("true->show password")
            self.password.password = False
        else:
            print("false->hide password")
            self.password.password = True

    def clear_login(self):
        self.username.text = ""
        self.password.text = ""
        self.error_message.text = ""

    def check_login(self):
        print('check_login function')
        db_connection.db_connect()

        connection_status = db_connection.check_login(self.username.text, self.password.text)
        if connection_status:
            print("!!", connection_status)
            # sm.current = "after_login"
            self.parent.current = 'after_login'
            # TODO correct set_username here
            self.set_username()
        else:
            self.parent.current = 'login'
            if not self.username.text or not self.password.text:
                self.error_message.text = _("Please fill out all mandatory fields!")
                self.error_message.color = [1, 0, 0, 1]

            else:
                self.error_message.text = _("Please enter some valid credentials!")
                self.error_message.color = [1, 0, 0, 1]

    # functions used to make Enter work (to press button)
    def __init__(self, **kwargs):
        super(Login, self).__init__(**kwargs)
        Window.bind(on_key_down=self._on_keyboard_down)

    login_button = ObjectProperty(None)

    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        if self.login_button.focus and keycode == 40:  # 40 - Enter key pressed
            self.check_login()
