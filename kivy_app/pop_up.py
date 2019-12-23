from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
import kivy_app.parameters_popup as parameters
import kivy_app.help_popup as help
import kivy_app.sensors_popup as sensors
import kivy_app.feedback_popup as feedback


# contact.kv

class ContactPopUp(FloatLayout):
    pass

    @staticmethod
    def close_popup():
        print("closing popup")
        if isinstance(App.get_running_app().root_window.children[0], Popup):
            App.get_running_app().root_window.children[0].dismiss()

    @staticmethod
    def show_popup(title_name):
        if title_name == 'Contact the developer of the application':
            show = ContactPopUp()
            width = 400
            height = 400
        elif title_name == "Parameters":
            show = parameters.ParametersPopUp()
            width = App.get_running_app().root.width
            height = App.get_running_app().root.height
        elif title_name == "Help":
            show = help.HelpPopUp()
            width = 500
            height = 500
        elif title_name == "Sensors":
            show = sensors.SensorsPopUp()
            width = 580
            height = 500
        elif title_name == "Feedback":
            show = feedback.FeedbackPopUp()
            width = 580
            height = 500

        contact_popup = Popup(title=title_name, content=show, size_hint=(None, None),
                              size=(width, height)
                              # ,
                              # background='atlas://data/images/defaulttheme/bubble'
                              )
        contact_popup.open()
