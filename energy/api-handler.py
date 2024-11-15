import requests
import os
from dotenv import load_dotenv

class ApiHandler():
    def __init__(self):
        load_dotenv()
        self.token = os.getenv('ENERGY_API_TOKEN')
        self.zone = 'DE'

    def create_headers(self):
        return { 'auth-token': self.token }

    def authorize(self):
        url = f'https://api.electricitymap.org/v3/carbon-intensity/latest?zone={self.zone}'
        headers = self.create_headers()

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(data)

    def get_carbon_history(self):
        url = f'https://api.electricitymap.org/v3/carbon-intensity/history?zone={self.zone}'
        headers = self.create_headers()

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print(data)

api = ApiHandler()
api.get_carbon_history()