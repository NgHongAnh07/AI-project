import matplotlib.pyplot as plt
import re

raw_data = """
Epoch 1/100 | Train Loss: 0.6533 | Val Loss: 0.2048
Epoch 2/100 | Train Loss: 0.5626 | Val Loss: 0.1963
Epoch 3/100 | Train Loss: 0.2322 | Val Loss: 0.1963
Epoch 4/100 | Train Loss: 0.2237 | Val Loss: 0.2249
Epoch 5/100 | Train Loss: 0.2211 | Val Loss: 0.2101
Epoch 6/100 | Train Loss: 0.2261 | Val Loss: 0.2258
Epoch 7/100 | Train Loss: 0.2348 | Val Loss: 0.2058
Epoch 8/100 | Train Loss: 0.2223 | Val Loss: 0.2299
Epoch 9/100 | Train Loss: 0.2325 | Val Loss: 0.1988
Epoch 10/100 | Train Loss: 0.2319 | Val Loss: 0.2086
Epoch 11/100 | Train Loss: 0.2234 | Val Loss: 0.2378
Epoch 12/100 | Train Loss: 0.2460 | Val Loss: 0.2175
Epoch 13/100 | Train Loss: 0.2375 | Val Loss: 0.2319
Epoch 14/100 | Train Loss: 0.2337 | Val Loss: 0.2266
Epoch 15/100 | Train Loss: 0.2403 | Val Loss: 0.2251
Epoch 16/100 | Train Loss: 0.2366 | Val Loss: 0.2337
Epoch 17/100 | Train Loss: 0.2489 | Val Loss: 0.2188
Epoch 18/100 | Train Loss: 0.2434 | Val Loss: 0.3838
Epoch 19/100 | Train Loss: 0.7390 | Val Loss: 0.3284
Epoch 20/100 | Train Loss: 0.3467 | Val Loss: 0.3257
Epoch 21/100 | Train Loss: 0.3038 | Val Loss: 0.3040
Epoch 22/100 | Train Loss: 0.3082 | Val Loss: 0.3765
Epoch 23/100 | Train Loss: 0.3597 | Val Loss: 0.3393
Epoch 24/100 | Train Loss: 0.3170 | Val Loss: 0.2532
Epoch 25/100 | Train Loss: 0.2737 | Val Loss: 0.2570
Epoch 26/100 | Train Loss: 0.2635 | Val Loss: 0.2381
Epoch 27/100 | Train Loss: 0.2446 | Val Loss: 0.2350
Epoch 28/100 | Train Loss: 0.2418 | Val Loss: 0.2440
Epoch 29/100 | Train Loss: 0.2497 | Val Loss: 0.2173
Epoch 30/100 | Train Loss: 0.2464 | Val Loss: 0.2353
Epoch 31/100 | Train Loss: 0.2476 | Val Loss: 0.2369
Epoch 32/100 | Train Loss: 0.2498 | Val Loss: 0.2349
Epoch 33/100 | Train Loss: 0.2498 | Val Loss: 0.2376
Epoch 34/100 | Train Loss: 0.2437 | Val Loss: 0.2301
Epoch 35/100 | Train Loss: 0.2490 | Val Loss: 0.2321
Epoch 36/100 | Train Loss: 0.2450 | Val Loss: 0.2212
Epoch 37/100 | Train Loss: 0.2267 | Val Loss: 0.2397
Epoch 38/100 | Train Loss: 0.2323 | Val Loss: 0.2239
Epoch 39/100 | Train Loss: 0.2324 | Val Loss: 0.2171
Epoch 40/100 | Train Loss: 0.2326 | Val Loss: 0.2209
Epoch 41/100 | Train Loss: 0.2320 | Val Loss: 0.2255
Epoch 42/100 | Train Loss: 0.2294 | Val Loss: 0.2596
Epoch 43/100 | Train Loss: 0.2356 | Val Loss: 0.1786
Epoch 44/100 | Train Loss: 0.2032 | Val Loss: 0.1850
Epoch 45/100 | Train Loss: 0.2022 | Val Loss: 0.1747
Epoch 46/100 | Train Loss: 0.2015 | Val Loss: 0.1892
Epoch 47/100 | Train Loss: 0.2058 | Val Loss: 0.1811
Epoch 48/100 | Train Loss: 0.2019 | Val Loss: 0.1808
Epoch 49/100 | Train Loss: 0.1914 | Val Loss: 0.1787
Epoch 50/100 | Train Loss: 0.1946 | Val Loss: 0.1756
Epoch 51/100 | Train Loss: 0.2037 | Val Loss: 0.1814
Epoch 52/100 | Train Loss: 0.2133 | Val Loss: 0.1804
Epoch 53/100 | Train Loss: 0.2071 | Val Loss: 0.1750
Epoch 54/100 | Train Loss: 0.1958 | Val Loss: 0.1746
Epoch 55/100 | Train Loss: 0.2072 | Val Loss: 0.1938
Epoch 56/100 | Train Loss: 0.2072 | Val Loss: 0.1784
Epoch 57/100 | Train Loss: 0.1984 | Val Loss: 0.1745
Epoch 58/100 | Train Loss: 0.2002 | Val Loss: 0.1802
Epoch 59/100 | Train Loss: 0.1981 | Val Loss: 0.1765
Epoch 60/100 | Train Loss: 0.2037 | Val Loss: 0.1757
Epoch 61/100 | Train Loss: 0.2026 | Val Loss: 0.1958
Epoch 62/100 | Train Loss: 0.2058 | Val Loss: 0.1955
Epoch 63/100 | Train Loss: 0.2058 | Val Loss: 0.1938
Epoch 64/100 | Train Loss: 0.2143 | Val Loss: 0.1932
Epoch 65/100 | Train Loss: 0.2078 | Val Loss: 0.1981
Epoch 66/100 | Train Loss: 0.2058 | Val Loss: 0.2001
Epoch 67/100 | Train Loss: 0.2130 | Val Loss: 0.2041
Epoch 68/100 | Train Loss: 0.2156 | Val Loss: 0.1976
Epoch 69/100 | Train Loss: 0.2064 | Val Loss: 0.2002
Epoch 70/100 | Train Loss: 0.2090 | Val Loss: 0.2145
Epoch 71/100 | Train Loss: 0.2116 | Val Loss: 0.1947
Epoch 72/100 | Train Loss: 0.2049 | Val Loss: 0.1952
Epoch 73/100 | Train Loss: 0.2008 | Val Loss: 0.2174
Epoch 74/100 | Train Loss: 0.2161 | Val Loss: 0.2034
Epoch 75/100 | Train Loss: 0.2114 | Val Loss: 0.2121
Epoch 76/100 | Train Loss: 0.2123 | Val Loss: 0.2047
Epoch 77/100 | Train Loss: 0.2217 | Val Loss: 0.2189
Epoch 78/100 | Train Loss: 0.2228 | Val Loss: 0.2165
Epoch 79/100 | Train Loss: 0.2191 | Val Loss: 0.1977
Epoch 80/100 | Train Loss: 0.2040 | Val Loss: 0.1852
Epoch 81/100 | Train Loss: 0.2078 | Val Loss: 0.2045
Epoch 82/100 | Train Loss: 0.2186 | Val Loss: 0.2022
Epoch 83/100 | Train Loss: 0.2107 | Val Loss: 0.2045
Epoch 84/100 | Train Loss: 0.2107 | Val Loss: 0.2091
Epoch 85/100 | Train Loss: 0.2185 | Val Loss: 0.2024
Epoch 86/100 | Train Loss: 0.2087 | Val Loss: 0.2019
Epoch 87/100 | Train Loss: 0.2079 | Val Loss: 0.1923
Epoch 88/100 | Train Loss: 0.2070 | Val Loss: 0.1984
Epoch 89/100 | Train Loss: 0.2072 | Val Loss: 0.2031
Epoch 90/100 | Train Loss: 0.2136 | Val Loss: 0.2067
Epoch 91/100 | Train Loss: 0.2158 | Val Loss: 0.2126
Epoch 92/100 | Train Loss: 0.2155 | Val Loss: 0.2164
Epoch 93/100 | Train Loss: 0.2117 | Val Loss: 0.1912
Epoch 94/100 | Train Loss: 0.2109 | Val Loss: 0.2145
Epoch 95/100 | Train Loss: 0.2096 | Val Loss: 0.1920
Epoch 96/100 | Train Loss: 0.2120 | Val Loss: 0.2225
Epoch 97/100 | Train Loss: 0.2303 | Val Loss: 0.2115
Epoch 98/100 | Train Loss: 0.2239 | Val Loss: 0.2081
Epoch 99/100 | Train Loss: 0.2249 | Val Loss: 0.2072
Epoch 100/100 | Train Loss: 0.2013 | Val Loss: 0.2457
"""

def plot_learning_curve(data):
    epochs = [int(x) for x in re.findall(r'Epoch\s+(\d+)/100', data)]
    train_loss = [float(x) for x in re.findall(r'Train Loss:\s+([\d.]+)', data)]
    val_loss = [float(x) for x in re.findall(r'Val Loss:\s+([\d.]+)', data)]

    if not epochs:
        print("Error: No data found. Check your raw_data format.")
        return

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, label='Training Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='#d62728', linewidth=2, linestyle='--')

    plt.title('Hardware Tracking Project: Learning Curve (100 Epochs)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Epochs', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')

    if len(epochs) >= 19:
        plt.annotate('Stochastic Variation', xy=(19, 0.7390), xytext=(35, 0.75),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.tight_layout()
    plt.savefig('learning_curve_final.png', dpi=300)
    print(f"Success! {len(epochs)} epochs processed. 'learning_curve_final.png' has been created.")

if __name__ == "__main__":
    plot_learning_curve(raw_data)