import os
import eodhd

def lookup_index_names(index_tickers, api_key=None):
    """
    Look up the correct names of indexes in EODHD by their tickers using the EODHD library.
    
    Args:
        index_tickers (dict): Dictionary mapping index names to potential tickers
        api_key (str): EODHD API key
    
    Returns:
        dict: Information about each index with corrected tickers and names
    """
    if api_key is None:
        api_key = os.environ.get('EODHD_API_KEY')
        if not api_key:
            raise ValueError("EODHD API key not provided and not found in environment variables")
    
    # Initialize the EODHD API client
    api = eodhd.APIClient(api_key)
    results = {}
    
    for name, ticker in index_tickers.items():
        # Try different variations of the ticker
        variations = [
            ticker,
            ticker.replace('^', ''),
            ticker.replace('.US', '.INDX'),
            ticker.split('.')[0].replace('^', '')
        ]
        
        found = False
        for variation in variations:
            if not variation:
                continue
                
            try:
                # Search for the index using the EODHD library
                search_results = api.search(variation, search_type='index')
                
                if search_results and len(search_results) > 0:
                    data = search_results[0]  # Take first result
                    results[name] = {
                        "original_ticker": ticker,
                        "eodhd_ticker": data.get("Code"),
                        "eodhd_name": data.get("Name")
                    }
                    found = True
                    break
            except Exception as e:
                print(f"Error searching for {variation}: {e}")
                continue
        
        if not found:
            results[name] = {
                "original_ticker": ticker,
                "eodhd_ticker": None,
                "eodhd_name": "Not found"
            }
    
    return results

def main():
    # Define the indexes to look up
    indexes = {
        'S&P 500': '^GSPC.US',
        'Dow Jones Industrial Average': 'DJI.INDX',
        'Russell 2000': 'RUT.INDX',
        'CBOE Volatility Index': 'VIX.INDX'
    }
    
    # Get API key
    api_key = os.environ.get('EODHD_API_KEY')
    if not api_key:
        print("Please set the EODHD_API_KEY environment variable")
        return
    
    # Lookup index names
    results = lookup_index_names(indexes, api_key)
    
    # Print results
    print("Index Lookup Results:")
    print("-" * 60)
    for name, info in results.items():
        print(f"{name}:")
        print(f"  Original ticker: {info['original_ticker']}")
        print(f"  EODHD ticker: {info['eodhd_ticker'] or 'Not found'}")
        print(f"  EODHD name: {info['eodhd_name']}")
        print("-" * 60)

if __name__ == "__main__":
    main()