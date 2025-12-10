# Deployment Guide

## Fixing README.md Showing Instead of Site

If you're seeing the README.md file instead of your Docusaurus site on GitHub Pages, follow these steps:

### 1. Verify Repository Name

Check your repository name. The `baseUrl` in `docusaurus.config.js` must match:

- Repository: `type1compute/docs` → `baseUrl: '/docs/'`
- Repository: `type1compute/pages` → `baseUrl: '/pages/'`
- Repository: `type1compute/type1compute.github.io` → `baseUrl: '/'` (user/organization site)

### 2. Configure GitHub Pages Settings

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select:
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
4. Click **Save**

### 3. Deploy Using GitHub Actions (Recommended)

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that automatically:
- Builds the site when you push to `main` or `master`
- Deploys to the `gh-pages` branch
- Handles the `.nojekyll` file automatically

**To use this:**
1. Push your code to the `main` or `master` branch
2. The workflow will run automatically
3. Check the **Actions** tab to see the deployment status
4. Wait a few minutes for GitHub Pages to update

### 4. Manual Deployment

If you prefer manual deployment:

```bash
# Build the site
npm run build

# Deploy to gh-pages branch
GIT_USER=type1compute npm run deploy
```

Or with SSH:

```bash
USE_SSH=true npm run deploy
```

### 5. Verify Deployment

After deployment:

1. Check the `gh-pages` branch exists and contains the `build` folder contents
2. Verify `.nojekyll` file is in the root of `gh-pages` branch
3. Wait 1-2 minutes for GitHub Pages to rebuild
4. Visit: `https://type1compute.github.io/docs/` (adjust URL based on your repo name)

### 6. Troubleshooting

**Issue: Still seeing README.md**
- ✅ Check `baseUrl` matches repository name
- ✅ Verify GitHub Pages is set to serve from `gh-pages` branch
- ✅ Ensure `.nojekyll` file exists in `gh-pages` branch root
- ✅ Clear browser cache and try incognito mode
- ✅ Wait a few minutes for GitHub Pages to update

**Issue: 404 errors**
- ✅ Check `baseUrl` is correct (should end with `/`)
- ✅ Verify all assets are in the `build` folder
- ✅ Check browser console for 404 errors on specific files

**Issue: Styles not loading**
- ✅ Check `baseUrl` is correct
- ✅ Verify all static assets are deployed
- ✅ Check network tab for failed resource loads

### 7. Current Configuration

```javascript
// docusaurus.config.js
url: 'https://type1compute.github.io',
baseUrl: '/docs/',
organizationName: 'type1compute',
projectName: 'docs',
```

If your repository name is different, update these values accordingly.
