# -*- coding: utf-8 -*-
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window
from user import mysql_conn as db_connection
from kniteditor.localization import _, change_language_to, current_language

from user.login import Login


class Account(Screen):
    pass

    @staticmethod
    def translate():
        Login.translate()  # change_language_to("en")

    @staticmethod
    def translate_ro():
        Login.translate_ro()  # change_language_to("ro")

    def show_hide_pass(self):
        from user.login import Login
        print("Account show_hide_pass")
        Login.show_hide_pass(self)  # use account class for param

    def clear_register_form(self):
        self.first_name.text = ""
        self.last_name.text = ""
        self.email.text = ""
        self.phone.text = ""
        self.username.text = ""
        self.password.text = ""

    def register_account(self):
        if not self.username.text or not self.password.text or not self.email.text:
            self.error_message.text = _("Please fill out all mandatory fields!")
        else:
            if not self.last_name.text:
                self.last_name.text = "NULL"
            if not self.first_name.text:
                self.first_name.text = "NULL"
            if not self.phone.text:
                self.phone.text = "NULL"
            db_connection.insert_values_user(self.username.text, self.password.text, self.email.text,
                                             self.first_name.text,
                                             self.last_name.text, self.phone.text)
            self.parent.current = "login"

        self.clear_register_form()

    # functions used to make Enter work (to press button)
    def __init__(self, **kwargs):
        super(Account, self).__init__(**kwargs)
        Window.bind(on_key_down=self._on_keyboard_down)

    register_button = ObjectProperty(None)

    def _on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        if self.register_button.focus and keycode == 40:  # 40 - Enter key pressed
            self.register_account()
