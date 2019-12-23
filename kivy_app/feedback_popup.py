from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
import kivy_app.pop_up as pop_up
import smtplib  # mail protocol


# feedback.kv
class FeedbackPopUp(BoxLayout):
    # TODO add checker on email+less secure apps

    def send_email(self):
        print("send e-mail to developer")
        sender_email = self.email.text
        sender_password = self.password.text
        sender_feedback = self.feedback_message.text
        print("email is: ", type(sender_email))
        print("pass is: ", sender_password)
        print("feedback message is: ", sender_feedback)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()  # encrypt connection
        server.ehlo()
        server.login(sender_email, sender_password)

        subject = 'Feedback'
        msg = sender_feedback

        msg = f"Subject: {subject} \n\n {msg}"
        server.sendmail(
            # from,to,message
            sender_email,
            'iuliana.stroia97@gmail.com',
            msg
        )
        print("E-MAIL HAS BEEN SENT!")
        server.quit()

    pass
