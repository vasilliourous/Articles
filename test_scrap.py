from scrapling import StealthyFetcher

url = "https://cybersecuritynews.com/hackers-hide-backdoor-in-trusted-wordpress-plugins/"

print(f"Testing fetch of: {url}\n")
try:
    page = StealthyFetcher.fetch(url, timeout=30000, network_idle=True)
    if page is None:
        print("❌ FAILED: StealthyFetcher returned None")
    else:
        html = page.get_all_text()
        if len(html) > 200:
            print(f"✅ SUCCESS — stealth browser works ({len(html)} chars)")
            print("\nFirst 300 characters:")
            print(html[:300])
        else:
            print(f"⚠️  Short content: {len(html)} chars")
except Exception as e:
    print(f"❌ CRASHED: {e}")
