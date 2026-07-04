from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException

import pandas as pd
import time

print("Opening browser...")

# Chrome options
options = webdriver.ChromeOptions()

# Block notifications
prefs = {
    "profile.default_content_setting_values.notifications": 2
}

options.add_experimental_option("prefs", prefs)

options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")

# Open Chrome
driver = webdriver.Chrome(options=options)

all_data = []

try:

    # Open page
    url = "https://merolagani.com/CompanyDetail.aspx?symbol=LICN"
    driver.get(url)

    time.sleep(5)

    # Open Price History
    price_history = driver.find_element(By.LINK_TEXT, "Price History")

    driver.execute_script(
        "arguments[0].scrollIntoView(true);",
        price_history
    )

    time.sleep(2)

    driver.execute_script(
        "arguments[0].click();",
        price_history
    )

    print("Opened Price History tab.")

    time.sleep(5)

    page = 1

    while True:

        print(f"\nReading page {page}...")

        # Close popup if exists
        try:
            alert = driver.switch_to.alert
            alert.dismiss()
            time.sleep(1)
        except:
            pass

        # Wait for table
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "table"))
        )

        rows_found = 0

        # Retry protection
        for attempt in range(3):

            try:

                # Re-fetch tables every time
                tables = driver.find_elements(By.TAG_NAME, "table")

                for table in tables:

                    rows = table.find_elements(By.TAG_NAME, "tr")

                    for row in rows[1:]:

                        cols = row.find_elements(By.TAG_NAME, "td")

                        if len(cols) >= 7:

                            rows_found += 1

                            all_data.append({
                                "Date": cols[1].text.strip(),
                                "Close": cols[2].text.strip(),
                                "Change": cols[3].text.strip(),
                                "High": cols[4].text.strip(),
                                "Low": cols[5].text.strip(),
                                "Open": cols[6].text.strip(),
                                "Volume": cols[7].text.strip()
                                if len(cols) > 7 else ""
                            })

                break

            except StaleElementReferenceException:

                print("Retrying table read...")
                time.sleep(2)

        print(f"Rows this page: {rows_found}")
        print(f"Total rows: {len(all_data)}")

        # Next page
        try:

            next_button = driver.find_element(By.LINK_TEXT, "Next")

            driver.execute_script(
                "arguments[0].scrollIntoView(true);",
                next_button
            )

            time.sleep(1)

            driver.execute_script(
                "arguments[0].click();",
                next_button
            )

            page += 1

            time.sleep(4)

        except Exception as e:

            print("\nFinished all pages.")
            print(e)

            break

finally:

    driver.quit()

# Save CSV
df = pd.DataFrame(all_data)

# Remove duplicates
df = df.drop_duplicates()

df.to_csv("licn_data.csv", index=False)

print("\nDONE!")
print(f"Total rows downloaded: {len(df)}")

print("\nFirst 5 rows:")
print(df.head())