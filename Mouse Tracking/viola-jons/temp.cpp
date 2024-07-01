#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
#define F first
#define S second
const int N = 5e3 + 10, mod = 1e9 + 7;
int dp[N][N];
vector<int> v, cnt;
vector<pair<int, int>> valid;
int goDp(int i, int idx = 1, int before = 0)
{
    if (i >= valid.size())
        return 0;
    int &ans = dp[i][idx];
    if (~ans)
        return ans;
    int x = valid[i].S;
    ans = goDp(i + 1, idx, before);
    if (idx + cnt[x] + before <= valid[i].F)
        ans = max(ans, 1 + goDp(i + 1, idx + cnt[x], before + 1));
    return ans;
}

void solve(int tc)
{
    int n;
    cin >> n;
    v = cnt = vector<int>(n + 1);
    set<int> nums;
    for (int i = 0; i < n; i++)
    {
        cin >> v[i];
        nums.insert(v[i]);
        cnt[v[i]]++;
    }
    valid.resize(0);
    vector<int> dist(nums.begin(), nums.end());
    for (int i = 1; i < dist.size(); i++)
    {
        valid.push_back({i, dist[i]});
    }
    for (int i = 0; i <= n; i++)
        for (int j = 0; j <= n; j++)
            dp[i][j] = -1;
    int ret = goDp(0, 0);
    int ans = (int)dist.size() - ret;
    cout << ans;
}

int main()
{
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr), cout.tie(nullptr);
    // freopen("in.txt", "r", stdin);
    // freopen("out.txt", "w", stdout);
    int t = 1;
    cin >> t;
    for (int tc = 1; tc <= t; tc++)
    {
        solve(tc);
        cout << '\n';
    }
    return 0;
}