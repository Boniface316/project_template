from __future__ import annotations

import typing as T
from .base import Service

import pync
from abc import abstractmethod
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class AlertsService(Service):
    """Service for sending notifications.

    Require libnotify-bin on Linux systems.

    In production, use with Slack, Discord, or emails.

    https://plyer.readthedocs.io/en/latest/api.html#plyer.facades.Notification

    Parameters:
        enable (bool): use notifications or print.
        app_name (str): name of the application.
        timeout (int | None): timeout in secs.
    """

    enable: bool = True
    app_name: str = "{{cookiecutter.name}}"
    timeout: int | None = None

    @T.override
    def start(self) -> None:
        pass

    @abstractmethod
    def notify(self, title: str, message: str) -> None:
        """Send a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        raise NotImplementedError

    def _print(self, title: str, message: str) -> None:
        """Print a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        print(f"[{self.app_name}] {title}: {message}")


class DesktopAlertServices(AlertsService):
    """Service for sending desktop notifications.

    Use plyer to send desktop notifications.

    https://plyer.readthedocs.io/en/latest/api.html#plyer.facades.Notification

    Parameters:
        enable (bool): use notifications or print.
        app_name (str): name of the application.
        timeout (int | None): timeout in secs.
    """

    @T.override
    def notify(self, title: str, message: str) -> None:
        """Send a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        if self.enable:
            pync.notify(
                title=title,
                message=message,
                app_name=self.app_name,
                timeout=self.timeout,
            )
        else:
            self._print(title, message)


class SlackAlertServices(AlertsService):
    """Service for sending slack notifications.

    Use slack to send slack notifications.

    https://slack.dev/python-slack-sdk/web/index.html

    Parameters:
        enable (bool): use notifications or print.
        app_name (str): name of the application.
        timeout (int | None): timeout in secs.
    """

    @T.override
    def notify(self, title: str, message: str, channel: str) -> None:
        """Send a notification to the system.

        Args:
            title (str): title of the notification.
            message (str): message of the notification.
        """
        if self.enable:
            client = WebClient(token=os.getenv("SLACK_API_TOKEN"))

            try:
                response = client.chat_postMessage(
                    channel=f"#{channel}", text=f"{title}: {message}"
                )
            except SlackApiError as e:
                print(f"Error sending message: {e.response['error']}")
            pass
        else:
            self._print(title, message)
