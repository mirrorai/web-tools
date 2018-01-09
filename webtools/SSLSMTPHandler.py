# -*- coding:utf-8 -*-
import smtplib
from logging.handlers import SMTPHandler


# Provide a class to allow SSL (Not TLS) connection for mail handlers by overloading the emit() method
# http://stackoverflow.com/questions/36937461/how-can-i-send-an-email-using-python-loggings-smtphandler-and-ssl
class SSLSMTPHandler(SMTPHandler):
    def emit(self, record):
        """
        Emit a record.
        """
        # noinspection PyBroadException
        try:
            port = self.mailport
            if not port:
                port = smtplib.SMTP_PORT
            smtp = smtplib.SMTP_SSL(self.mailhost, port)
            msg = self.format(record)
            if self.username:
                smtp.login(self.username, self.password)
            smtp.sendmail(self.fromaddr, self.toaddrs, 'Subject: %s\n\n%s' % (self.subject, msg))
            smtp.quit()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
