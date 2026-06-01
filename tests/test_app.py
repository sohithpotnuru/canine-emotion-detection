import unittest

from app import app


class AppProductionSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_root_returns_200_even_when_models_are_missing(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "degraded")

    def test_health_endpoint_reports_service_status(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["service"], "dog-emotion-api")


if __name__ == "__main__":
    unittest.main()
