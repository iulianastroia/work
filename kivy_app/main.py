from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from googletrans import Translator


class Menu(FloatLayout):
    # get id from my.kv
    account = ObjectProperty(None)
    create_account = ObjectProperty(None)
    login = ObjectProperty(None)
    dashboard = ObjectProperty(None)
    settings = ObjectProperty(None)
    alert = ObjectProperty(None)
    language = ObjectProperty(None)
    english = ObjectProperty(None)
    romanian = ObjectProperty(None)
    about = ObjectProperty(None)
    contact = ObjectProperty(None)
    help = ObjectProperty(None)
    sensors_used = ObjectProperty(None)
    feedback = ObjectProperty(None)
    parameters = ObjectProperty(None)

    # boolean function to check needed language
    def check_language(self, instance):
        if instance.text == "Engleză":
            boolean_value = True
            return boolean_value
        else:
            boolean_value = False
            return boolean_value

    # translate from ro->en or en->ro
    def translate(self, instance):
        # get text from menu button into a list
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
            print("key of dictionary:  ", key)

            # translate to english
            if boolean_value:
                self.ids[key].text = translator.translate(id_text_dict.get(key),
                                                          dest='en').text
            else:
                # translate to romanian
                if instance.text == "Romanian":
                    self.ids[key].text = translator.translate(id_text_dict.get(key),
                                                              dest='ro').text


class MyApp(App):
    def build(self):
        return Menu()


if __name__ == "__main__":
    MyApp().run()
