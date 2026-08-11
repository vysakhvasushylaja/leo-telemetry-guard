"""
ntfy Real-Time Alerting Integration
--------------------------------------
Sends push notifications via a self-hosted ntfy server whenever a
significant zero-trust security event occurs: a node is quarantined,
re-authenticated, or a tampering attempt is detected.
"""

import requests

NTFY_TOPIC = "leo-telemetry-vyshu2026"
NTFY_URL = f"http://192.168.200.9:9090/{NTFY_TOPIC}"


def send_alert(title: str, message: str, priority: str = "default", tags: str = ""):
    try:
        response = requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags,
            },
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"[ntfy] Failed to send alert: {e}")
        return False


def alert_quarantine(node_id: str, trust_score: int, packet_idx: int):
    send_alert(
        title=f"Node Quarantined: {node_id}",
        message=(f"{node_id} dropped below trust threshold at packet {packet_idx} "
                  f"(score: {trust_score}). Node isolated from network."),
        priority="urgent",
        tags="rotating_light,satellite"
    )


def alert_reauth(node_id: str, trust_score: int, packet_idx: int):
    send_alert(
        title=f"Node Re-authenticated: {node_id}",
        message=(f"{node_id} passed re-authentication at packet {packet_idx} "
                  f"(new trust score: {trust_score}). Node restored to network."),
        priority="default",
        tags="white_check_mark,satellite"
    )


def alert_tamper_detected(node_id: str, packet_idx: int):
    send_alert(
        title=f"Tampering Detected: {node_id}",
        message=f"Signature verification failed for packet {packet_idx} from {node_id}.",
        priority="high",
        tags="warning"
    )


if __name__ == "__main__":
    print(f"Sending test alert to topic: {NTFY_TOPIC}")
    print(f"Subscribe at: {NTFY_URL}")
    ok = send_alert(
        title="LEO Telemetry Guard - Test Alert",
        message="Zero-trust alerting system is online and connected.",
        priority="default",
        tags="satellite"
    )
    print("Alert sent successfully!" if ok else "Alert failed to send.")
