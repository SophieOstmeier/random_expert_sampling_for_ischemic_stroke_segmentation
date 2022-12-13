from typing import Final, final

@final
class config_base:

    send_email: Final = True

    notif_email_address: Final = "sophieemailsender@gmail.com"
    notif_email_password: Final = "azjemgarcfebevah"
    notif_targets: Final = ["ostmeiersophie@gmail.com"]

    def __init__(self):
        self.threshold_do_not_use = None


if __name__ != "__main__":
    pass

    cb = config_base()
    print("Test", cb.send_email)

