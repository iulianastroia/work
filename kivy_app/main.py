from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from googletrans import Translator
import kivy_app.pop_up as popup
from kivy.lang import Builder

# load "another" kivy file
Builder.load_file('contact.kv')
Builder.load_file('parameters.kv')
Builder.load_file('help.kv')
Builder.load_file('sensors.kv')
Builder.load_file('feedback.kv')


class Menu(FloatLayout):
    # boolean function to check needed language
    @staticmethod
    def check_language(instance):
        if instance.text == "Engleză":
            boolean_value = True
            return boolean_value
        else:
            boolean_value = False
            return boolean_value

    def id_list(self):
        # get text from button
        button_text = [
            self.ids['account'].text,
            self.ids['create_account'].text,
            self.ids['login'].text,
            self.ids['dashboard'].text,
            self.ids['settings'].text,
            self.ids['alert'].text,
            self.ids['language'].text,
            self.ids['english'].text,
            self.ids['romanian'].text,
            self.ids['about'].text,
            self.ids['contact'].text,
            self.ids['help'].text,
            self.ids['sensors_used'].text,
            self.ids['feedback'].text,
            self.ids['parameters'].text
        ]
        return button_text

    # translate from ro->en or en->ro
    def translate(self, instance):
        button_text = self.id_list()

        # create connection between id(keys) and text(values) from buttons
        id_text_dict = dict(zip(self.ids.keys(), button_text))
        translator = Translator()

        # print dictionary key:value
        print("dictionary is: ", id_text_dict)

        # boolean_value to check needed language
        boolean_value = self.check_language(instance)
        print("boolean_value from check_language function is: ", boolean_value)
        for key in self.ids.keys():
            # print key of dictionary
            print("key of dictionary:", key)

            # translate to english
            if boolean_value:
                self.ids[key].text = translator.translate(id_text_dict.get(key),
                                                          dest='en').text
            else:
                # translate to romanian
                if instance.text == "Romanian":
                    self.ids[key].text = translator.translate(id_text_dict.get(key),
                                                              dest='ro').text

    def display_popup(self, instance):
        if instance in self.ids.values():
            button_id = list(self.ids.keys())[list(self.ids.values()).index(instance)]
            print("pressed button id is ", button_id)
            if button_id == "contact":
                popup.ContactPopUp.show_popup("Contact the developer of the application")
            elif button_id == "parameters":
                popup.ContactPopUp.show_popup("Parameters")
            elif button_id == "help":
                popup.ContactPopUp.show_popup("Help")
            elif button_id == "sensors_used":
                popup.ContactPopUp.show_popup("Sensors")
            elif button_id == "feedback":
                popup.ContactPopUp.show_popup("Feedback")


class MyApp(App):
    def build(self):
        self.title = 'Air Pollution'
        from kivy.core.window import Window
        Window.size = (1000, 600)
        # Window.fullscreen = 'auto' fullscreen complet

        return Menu()


if __name__ == "__main__":
    # Config.set('graphics', 'window_state', 'maximized') good->maximise

    MyApp().run()
