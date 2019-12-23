from kivy.uix.anchorlayout import AnchorLayout

import kivy_app.pop_up as pop_up


# help.kv
class HelpPopUp(AnchorLayout):
    @staticmethod
    def close_popup():
        pop_up.ContactPopUp.close_popup()

    pass
