from langchain_core.tools import tool
import requests


@tool
def get_bwen_market_data():
    """
    Get cryptocurrency market data about BWEN such as market cap or price using dexscreener API.
    You will also use this to retrieve the social media links.
    """
    try:
        response = requests.get("https://api.dexscreener.com/latest/dex/tokens/7pmuGLLYdJ2mc7chZwEJAaxuWALAYqaVqbUwzzyHcA7D")
        if response.status_code == 200:
            data = response.json()
            if data and 'pairs' in data and data['pairs']:
                coin = data['pairs'][0]  # Access the first pair

                # Extract relevant information from baseToken and quoteToken
                base_token = coin['baseToken']
                quote_token = coin['quoteToken']
                price = float(coin['priceUsd'])  # Use priceUsd for the price

                # Format price based on value
                if price < 0.01:
                    formatted_price = f"${price:.8f}"
                elif price < 1:
                    formatted_price = f"${price:.4f}"
                else:
                    formatted_price = f"${price:,.2f}"

                result = f"""
{base_token['name']} ({base_token['symbol'].upper()}) Market Data:
Price: {formatted_price}
Market Cap: ${coin['marketCap']:,.0f}
Liquidity: ${coin['liquidity']['usd']:,.0f}
24h Volume: ${coin['volume']['h24']:,.2f}
24h Change: {coin['priceChange']['h24']:.2f}%
Transactions (Last 24h): Buys: {coin['txns']['h24']['buys']}, Sells: {coin['txns']['h24']['sells']}
Price Change (Last 1h): {coin['priceChange']['h1']:.2f}%
Price Change (Last 6h): {coin['priceChange']['h6']:.2f}%
Price Change (Last 5m): {coin['priceChange']['m5']:.2f}%
Image URL: {coin['info']['imageUrl']}
Website: {coin['info']['websites'][0]['url'] if coin['info']['websites'] else 'N/A'}
Social Links: 
"""
                for social in coin['info']['socials']:
                    result += f"- {social['type'].capitalize()}: {social['url']}\n"

                return result.strip()

    except Exception as e:
        return f"Error fetching market data: {str(e)}"

    return "Could not find market data"
