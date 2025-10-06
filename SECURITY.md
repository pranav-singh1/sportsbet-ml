# 🔒 Security Guide

## ⚠️ CRITICAL: API Key Security

### Your API keys are now secure! ✅

**What I fixed:**
1. ✅ Created `.env` file for your API keys (NOT tracked in git)
2. ✅ Created `.env.example` template (safe to share)
3. ✅ Updated all code to use environment variables
4. ✅ Verified `.gitignore` blocks `.env` from being committed

---

## 🚨 IMPORTANT: Remove Keys from Git History

**Your API key is currently in your GitHub history!** Even though we've removed it from current files, it's still in previous commits.

### Option 1: Revoke and Replace (Recommended)

1. **Get a new API key:**
   - Go to https://the-odds-api.com/
   - Generate a new API key
   - Revoke the old one

2. **Update your .env file:**
   ```bash
   # Edit .env with your NEW key
   ODDS_API_KEY=your_new_api_key_here
   ```

### Option 2: Remove from Git History (Advanced)

```bash
# WARNING: This rewrites git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/config.py" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (be careful!)
git push origin --force --all
```

⚠️ **Better to just get a new API key!**

---

## 🔐 How to Use .env Files

### Setup (One Time)

1. **Copy the example:**
   ```bash
   cp .env.example .env
   ```

2. **Add your keys to .env:**
   ```bash
   # Edit .env
   ODDS_API_KEY=34ed2dc9566ad15743e1ef7eac40a2ca
   ```

3. **Never commit .env:**
   ```bash
   # Already in .gitignore, so it won't be tracked
   ```

### Using Keys in Code

**✅ CORRECT:**
```python
from config.config import ODDS_API_KEY

client = OddsAPIClient(ODDS_API_KEY)
```

**❌ NEVER DO THIS:**
```python
# NEVER hardcode keys!
client = OddsAPIClient('34ed2dc9566ad15743e1ef7eac40a2ca')
```

---

## 📋 Security Checklist

- [x] `.env` file created with API keys
- [x] `.env` in `.gitignore`
- [x] Code uses environment variables
- [x] `.env.example` template provided
- [ ] **You need to:** Get new API key and update .env
- [ ] **You need to:** Revoke old exposed key

---

## 🔍 Check What's Exposed

```bash
# Search for any hardcoded keys
grep -r "api_key\|API_KEY" --include="*.py" --include="*.ipynb"

# Check git history
git log -p | grep -i "api"
```

---

## 🛡️ Best Practices

### ✅ DO:
- Store keys in `.env` file
- Use environment variables
- Share `.env.example` template
- Revoke compromised keys immediately
- Use different keys for dev/prod

### ❌ DON'T:
- Commit `.env` to git
- Hardcode keys in code
- Share keys in Slack/Discord
- Use same key across projects
- Store keys in notebook outputs

---

## 🔄 If You Accidentally Commit a Key

1. **Immediately revoke the key** (don't wait!)
2. Generate a new key
3. Update `.env` with new key
4. Remove from git history (or just revoke and move on)

---

## 📞 Getting New API Keys

### The Odds API
1. Go to: https://the-odds-api.com/
2. Sign up / Log in
3. Dashboard → API Keys
4. Generate new key
5. Copy to `.env` file

### NBA API
- No API key needed!
- Just rate-limited, so use delays

---

## 🎯 Current Status

**Your .env file location:**
```
/Users/pranavsingh/sportsbet-ml/.env
```

**Contains:**
- ODDS_API_KEY (your private key)

**Safe to share:**
- .env.example (template only)
- All .py files (now use env variables)

**Action needed:**
1. Get new API key from OddsAPI
2. Update .env with new key
3. Revoke old exposed key

---

## ⚡ Quick Test

```bash
# Test that keys load correctly
python3 -c "from config.config import ODDS_API_KEY; print('✓ API key loaded' if ODDS_API_KEY else '✗ No key')"
```

**Expected output:** `✓ API key loaded`

---

## 🆘 Troubleshooting

**"ODDS_API_KEY not found!"**
```bash
# Make sure .env exists
ls -la .env

# Check it has content
cat .env

# Should show: ODDS_API_KEY=your_key_here
```

**"Module 'dotenv' not found"**
```bash
pip install python-dotenv
```

---

## 📚 Learn More

- [GitHub: Remove sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [The Odds API Docs](https://the-odds-api.com/liveapi/guides/v4/)
- [Python-dotenv](https://github.com/theskumar/python-dotenv)
