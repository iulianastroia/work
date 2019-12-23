from kivy.uix.boxlayout import BoxLayout
import kivy_app.pop_up as pop_up


# parameters.kv
class ParametersPopUp(BoxLayout):
    @staticmethod
    def close_popup():
        pop_up.ContactPopUp.close_popup()

    pass
