import lxml.html
import requests
import CountryNames

WIKIPEDIA_PREFIX = "https://en.wikipedia.org/wiki/"
BASE_QUERY = "//table[contains(@class, 'infobox')]/tbody"

class Person:
    def __init__(self, name):
        self.name = name

        url = WIKIPEDIA_PREFIX + name.replace(" ", "_")
        self.doc = lxml.html.fromstring(requests.get(url).content)
        self.country_names = CountryNames.get_country_names()
        self.date_of_birth = self.get_date_of_birth()
        self.birth_loc = self.get_birth_loc()

    def get_date_of_birth(self):
        xpath = BASE_QUERY + "/tr[th//text()='Born']//span[@class='bday']//text()"
        for elem in self.doc.xpath(xpath):
            return str(elem).replace(" ", "")

    def get_birth_loc(self):
        xpath = "//table[contains(@class, 'infobox')][1]/tbody/tr[th//text()='Born']//td[1]//text()"
        for elem in self.doc.xpath(xpath):
            potential_birth_loc = str(elem).replace(")", "").replace(")", "")
            for val in potential_birth_loc.split(","):
                val = val.strip().replace(" ", "_")
                if val not in self.country_names:
                    continue
                return val.replace(" ", "_")
