from langchain_core.tools import tool
import requests


@tool
def get_crypto_market_data(token: str = "bitcoin", currency: str = "usd") -> str:
    """
    Get cryptocurrency market data from CoinGecko.
    Examples: 
    - 'bitcoin' or 'btc' for market data in USD
    - 'ethereum eur' for market data in EUR
    """
    # Common aliases
    token_map = {
        "btc": "bitcoin",
        "eth": "ethereum",
        "sol": "solana",
        "matic": "polygon",
        "avax": "avalanche",
    }
    
    # Parse input
    parts = token.lower().split()
    token = parts[0]
    currency = parts[1].lower() if len(parts) > 1 else currency
    
    # Map common names to IDs
    token = token_map.get(token, token)
    
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": currency,
                "ids": token,
                "order": "market_cap_desc",
                "per_page": 1,
                "page": 1,
                "sparkline": False,
                "price_change_percentage": "24h,7d,30d"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            if data:
                coin = data[0]
                
                # Format price based on value
                price = coin['current_price']
                if price < 0.01:
                    formatted_price = f"${price:.8f}"
                elif price < 1:
                    formatted_price = f"${price:.4f}"
                else:
                    formatted_price = f"${price:,.2f}"
                
                result = f"""
{coin['name']} ({coin['symbol'].upper()}) Market Data:
Price: {formatted_price}
Market Cap Rank: #{coin['market_cap_rank']}
Market Cap: ${coin['market_cap']:,.0f}
24h Volume: ${coin['total_volume']:,.0f}
24h Change: {coin['price_change_percentage_24h']:.2f}%
7d Change: {coin['price_change_percentage_7d_in_currency']:.2f}%
30d Change: {coin['price_change_percentage_30d_in_currency']:.2f}%
24h High: ${coin['high_24h']:,.2f}
24h Low: ${coin['low_24h']:,.2f}
Circulating Supply: {coin['circulating_supply']:,.0f} {coin['symbol'].upper()}
"""
                return result.strip()
            
    except Exception as e:
        return f"Error fetching market data for {token}: {str(e)}"
    
    return f"Could not find market data for {token}" 
