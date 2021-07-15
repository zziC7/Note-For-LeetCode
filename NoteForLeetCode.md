# cNote of LeetCode 

### **2021.4.13** 二叉搜索树

> #### LeetCode 783-二叉搜索树节点最小距离 
>
> https://leetcode-cn.com/problems/minimum-distance-between-bst-nodes/
>
> #### ***二叉搜索树***
>
> ​	二叉搜索树(Binary Search Tree，简写BST)，又称为二叉排序树，属于树的一种，通过二叉树将数据组织起来，树的每个节点都包含了健值 key、数据值 data、左子节点指针、右子节点指针。其中键值 key 是最核心的部分，它的值决定了树的组织形状；数据值 data 是该节点对应的数据，有些场景可以忽略，举个例子，key 为身份证号而 data 为人名，通过身份证号找人名；左子节点指针指向左子节点；右子节点指针指向右子节点。
>
> #### ***二叉搜索树特点***
>
> - 左右子树也分别是二叉搜索树
> - 左子树的所有节点 key 值都小于它的根节点的 key 值
> - 右子树的所有节点 key 值都大于他的根节点的 key 值
> - 二叉搜索树可以为一棵空树
> - 一般来说，树中的每个节点的 key 值都不相等，但根据需要也可以将相同的 key 值插入树中

<img src="https://assets.leetcode.com/uploads/2021/02/05/bst1.jpg" alt="img"  />

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void BST2Vec(TreeNode* root, vector<int>& vec)
    {
        //将二叉搜索树转换为数组
        if(!root) return;
        BST2Vec(root->left,vec); //左
        vec.push_back(root->val); //中
        BST2Vec(root->right,vec); //右
    }

    int minDiffInBST(TreeNode* root) {
        vector<int> vec;
        BST2Vec(root,vec);
        int minDiff = INT_MAX;
        for(int i=0; i<vec.size()-1; ++i)
        {
            minDiff = min(minDiff,vec[i+1]-vec[i]);
        }
        return minDiff;
    }
};
```



### 2021.4.14 **前缀树(字典树) **

> #### **LeetCode 208-Trie **
>
> https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/fu-xue-ming-zhu-cong-er-cha-shu-shuo-qi-628gs/

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

```c++
class TrieNode{ 
    //前缀树节点
public:
    bool isEnd; //是否到一个串的末尾
    vector<TrieNode*> children;  //子节点
    TrieNode(): isEnd(false), children(26,nullptr){} //构造函数
    ~TrieNode(){  
        //析构
        for(auto& c:children)  delete c; //引用型自动推导，是指针传递而非值传递
    }
};

class Trie {
private:
    TrieNode* root; //根节点
public:
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* p = root;
        for(char ch : word){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) p->children[i] = new TrieNode(); //若这个子节点还没有创建，就new一个
            p = p->children[i]; //接着往下遍历
        }
        p->isEnd = true;  //遍历完一个字符串，在最后一个节点将isEnd设置为TRUE,代表一个字符串的结束
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode* p = root;
        for(char ch : word){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) return false; //没有这个字符的子节点，说明该字符串不存在
            p = p->children[i]; //接着往下遍历
        }
        return p->isEnd; //遍历完了，由最后这个节点的isEnd判断这是不是完整的字符串
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
         TrieNode* p = root;
        for(char ch : prefix){
            int i = ch - 'a'; //得到数组下标
            if(!p->children[i]) return false; //没有这个字符的子节点，说明由该字符串开头的字符串不存在
            p = p->children[i]; //接着往下遍历
        }
        return true; //遍历完了。与查找不同，不管isEnd是True还是False,都已经确定了有这个子串开头的字符串，所以返回true
    }
};
```



### 2021.4.15  动态规划 

> LeetCode 198-HouseRobber 
>
> https://leetcode-cn.com/problems/house-robber/solution/dong-tai-gui-hua-jie-ti-si-bu-zou-xiang-jie-cjavap/

动态规划的的四个解题步骤是：

1. 定义子问题
2. 写出子问题的递推关系
3. 确定 DP 数组的计算顺序
4. 空间优化（可选）

```c++
int rob(vector<int>& nums) {
    if (nums.size() == 0) {
        return 0;
    }
    // 子问题：
    // f(k) = 偷 [0..k) 房间中的最大金额

    // f(0) = 0
    // f(1) = nums[0]
    // f(k) = max{ rob(k-1), nums[k-1] + rob(k-2) }

    int N = nums.size();
    vector<int> dp(N+1, 0);
    dp[0] = 0;
    dp[1] = nums[0];
    for (int k = 2; k <= N; k++) {
        dp[k] = max(dp[k-1], nums[k-1] + dp[k-2]);
    }
    return dp[N];
}
```

空间优化： 空间复杂度： O(n) --> O(1)

```c++
int rob(vector<int>& nums) {
    int prev = 0;
    int curr = 0;

    // 每次循环，计算“偷到当前房子为止的最大金额”
    for (int i : nums) {
        // 循环开始时，curr 表示 dp[k-1]，prev 表示 dp[k-2]
        // dp[k] = max{ dp[k-1], dp[k-2] + i }
        int temp = max(curr, prev + i);
        prev = curr;
        curr = temp;
        // 循环结束时，curr 表示 dp[k]，prev 表示 dp[k-1]
    }

    return curr;
}
```

### 2021.4.18 快速排序

> https://blog.csdn.net/starzyh/article/details/90272347
>
> - ### 1.快排的实现逻辑：
>
>   - 先从数列中取出一个数作为基准数(通常取第一个数)。
>   - 分区过程，将比这个数大的数全放到它的右边，小于或等于它的数全放到它的左边。
>   - 再对左右区间重复第二步，直到各区间只有一个数。
>
>   ### 2.示意图
>
>   <img src="https://img-blog.csdnimg.cn/20190517164201640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N0YXJ6eWg=,size_16,color_FFFFFF,t_70" alt="img" style="zoom: 67%;" />
>
>   ### 3.代码
>
>   ```c++
>   void QuickSort(vector<int>& nums, int left, int right)
>   {
>   	if (left == right) return;
>   	if (left < right)
>   	{
>   		int pivot = nums[left];
>   		int low = left, high = right;
>   		while (low < high)
>   		{
>   			while (nums[high] >= pivot && low < high)
>   			{
>   				high--;
>   			}
>   			nums[low] = nums[high];
>                                                             
>   			while (nums[low] <= pivot && low < high)
>   			{
>   				low++;
>   			}
>   			nums[high] = nums[low];
>   		}
>   		nums[low] = pivot;
>   		QuickSort(nums, left, low - 1);
>   		QuickSort(nums, low + 1, right);
>   	}
>   }
>   ```



### 2021.4.19 哈希表

> https://blog.csdn.net/gp1330782530/article/details/106413778 unordered_set的介绍及使用
>
> https://leetcode-cn.com/problems/linked-list-cycle-ii/submissions/ LeetCode-142 环形链表
>
> 哈希表最大的优点，就是**把数据的存储和查找消耗的时间大大降低，几乎可以看成是常数时间；而代价仅仅是消耗比较多的内存**。然而在当前可利用内存越来越多的情况下，用空间换时间的做法是值得的。另外，编码比较容易也是它的特点之一。
>
> ```c++
>  ListNode *detectCycle(ListNode *head) {
>         unordered_set<ListNode*> visited; //哈希表
>         while(head){
>             if(visited.count(head))return head; //如果找到重复元素，说明有环，且这个节点就是环的起点
> 
>             visited.insert(head); //如果没有这个元素，就加入到哈希表里
>             head = head->next; // 继续向下遍历
>         }
>         return nullptr;
>     }
> ```
>
> 时间复杂度：O(N)，需要遍历链表中的每个节点
>
> 空间复杂度：O(N)，需要将每个节点存储到哈希表中
>
> *利用快慢指针可以将空间复杂度O(N) --> O(1) （追及问题） 这里略，详见LeetCode官方题解



### 2021.4.21  动态规划 

> LeetCode 91-解码方法
>
> https://leetcode-cn.com/problems/decode-ways/

```c++
int numDecodings(string s) {
	int n = s.size();
	vector<int> f(n + 1);  //代表长度为n的字符串的解码方法总数
	f[0] = 1; //初始条件：空字符串，一种解码方法
	for (int i = 0; i < n; ++i) {
		//判断末尾字符是否能解码（是不是0）
		if (s[i] != '0') {
			f[i + 1] += f[i];
		}
		//判断末尾2位是否能解码
		if (i > 0 && s[i - 1] != '0' && ((s[i - 1] - '0') * 10 + (s[i] - '0') <= 26)) {
			f[i + 1] += f[i - 1];
		}
	}
	return f[n];
}
```

时间复杂度：O(n)，遍历长度为n的字符串

空间复杂度：O(n),   创建大小为n+1的数组存放长度为n的字符串的解码方法总数



**空间优化**：O(n) --> O(1)注意到在状态转移方程中，f<sub>i</sub>的值仅与和 f<sub>i-1</sub>和f<sub>i-2</sub>有关，

因此我们可以使用三个变量进行状态转移，省去数组的空间。

```c++
int numDecodings(string s) {
	int n = s.size();
	// a = f[i-1], b = f[i], c = f[i+1]
	int a = 0, b = 1, c;
	//初始条件：空字符串，一种解码方法
	for (int i = 0; i < n; ++i) {
		c = 0; //每次进入循环都重置c为0
		//判断末尾字符是否能解码（是不是0）
		if (s[i] != '0') {
			c += b;
		}
		//判断末尾2位是否能解码
		if (i > 0 && s[i - 1] != '0' && ((s[i - 1] - '0') * 10 + (s[i] - '0') <= 26)) {
			c += a;
		}
        //往前挪一步
		a = b;
		b = c;
	}
	return c;
}
```



### 2021.4.24 动态规划

> LeetCode-377 组合总和Ⅳ https://leetcode-cn.com/problems/combination-sum-iv/

```c++
int combinationSum4(vector<int>& nums, int target) {
        vector<int> dp(target + 1);
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int& num : nums) {
                if (num <= i && dp[i - num] < INT_MAX - dp[i]) {
                    dp[i] += dp[i - num];
                }
            }
        }
        return dp[target];
    }
```



### 2021.4.25 二叉搜索树

> LeetCode-897 递增顺序搜索树 https://leetcode-cn.com/problems/increasing-order-search-tree/
>
> 给你一棵二叉搜索树，请你 **按中序遍历** 将其重新排列为一棵递增顺序搜索树，使树中最左边的节点成为树的根节点，并且每个节点没有左子节点，只有一个右子节点。

```c++
    void inorder(TreeNode *node, vector<int> &res) {
            if (node == nullptr) {
                return;
            }
            inorder(node->left, res);
            res.push_back(node->val);
            inorder(node->right, res);
        }

    TreeNode *increasingBST(TreeNode *root) {
        vector<int> res;
        inorder(root, res);

        TreeNode *dummyNode = new TreeNode(-1);
        TreeNode *currNode = dummyNode;
        for (int value : res) {
            currNode->right = new TreeNode(value);
            currNode = currNode->right;
        }
        return dummyNode->right;
    }
```



### 2021.4.26 二分查找的应用

> LeetCode-1011 在D天送达包裹的能力 https://leetcode-cn.com/problems/capacity-to-ship-packages-within-d-days/
>
> **二分查找算法的注意事项：**
>
> **1.循环退出条件：**
>
> 注意是low<=high
>
> `while(left <= right)` 的终止条件是 `left == right + 1`，写成区间的形式就是 `[right + 1, right]`，或者带个具体的数字进去 `[3, 2]`，可见这时候区间为空，因为没有数字既大于等于 3 又小于等于 2 的吧。所以这时候 while 循环终止是正确的，直接返回 -1 即可。
>
> `while(left < right)` 的终止条件是 `left == right`，写成区间的形式就是 `[left, right]`，或者带个具体的数字进去 `[2, 2]`，这时候区间非空，还有一个数 2，但此时 while 循环终止了。也就是说这区间 `[2, 2]` 被漏掉了，索引 2 没有被搜索，如果这时候直接返回 -1 就是错误的。
>
> **2.mid取值**
>
> 实际上，mid=(low+high)/2 这种写法是有问题的。因为如果 low 和 high 比较大的话，两者之和就有可能会溢出。改进的方法是将 mid 的计算方式写成low+(high-low)/2。更进一步，如果要将性能优化到极致的话，我们可以将这里的除以 2 操作转化成位运算 low+((high-low)>>1)。因为相比除法运算来说，计算机处理位运算要快得多。
>
> **3.low 和 high 的更新**
>
> low=mid+1，high=mid-1。注意这里的 +1 和 -1，如果直接写成 low=mid 或者 high=mid，就可能会发生死循环。比如，当 high=3，low=3 时，如果 a[3]不等于 value，就会导致一直循环不退出。

```c++
   class Solution {
public:
        bool check(vector<int> &weights, int D, int limit) {
        int cnt = 1; //记录该承载力下运送完所有包裹需要的天数
        int cur = 0; //记录现在这艘船这一天已经运送包裹的重量
        for(auto &weight : weights) {
            //如果运这个包裹会超载，那今天就运到这里，天数+1，重量清零
            if(cur + weight > limit) {
                cnt++;
                cur = 0;
            }
            if (cnt > D)return false; //当前天数已经超出要求，直接返回false
            cur += weight; //把当前这个包裹的重量记录到cur里
        }
        return cnt <= D; //运送完了，判断所需天数是否小于D
    }

    int shipWithinDays(vector<int>& weights, int D) {
        //二分查找
        int left = *max_element(weights.begin(), weights.end()); //最低运载能力：最大包裹的重量
        int right = accumulate(weights.begin(),weights.end(),0);//最大运载能力：所有包裹的重量和
        int ans = right;

        while(left<=right)
        {
            int mid = (left + right) /2;
            if(check(weights,D,mid))
            {
                right = mid - 1;
            }
            else left = mid + 1; //注意，这里一定return的是else里面的
        }
        return left;
    }
};
```



### 2021.4.27 二叉搜索树&dfs

> LeetCode-938 二叉搜索树范围和 https://leetcode-cn.com/problems/range-sum-of-bst/
>
> 当时做的时候没有多想，直接中序遍历转数组了，时间和空间效率都比较差

> #### 思路：
>
> **1、root 节点为空**
>
> 返回 0。
>
> **2、root 节点的值大于high**
>
> 由于二叉搜索树右子树上所有节点的值均大于根节点的值，即均大于high，故无需考虑右子树，返回左子树的范围和。
>
> **3、root 节点的值小于low**
>
> 由于二叉搜索树左子树上所有节点的值均小于根节点的值，即均小于low，故无需考虑左子树，返回右子树的范围和。
>
>  **4、root 节点的值在[low, high]范围内**
>
> 此时应返回 root 节点的值、左子树的范围和、右子树的范围和这三者之和。

```c++
class Solution {
public:
    int rangeSumBST(TreeNode *root, int low, int high) {
        if (root == nullptr) {
            return 0;
        }
        if (root->val > high) {
            return rangeSumBST(root->left, low, high);
        }
        if (root->val < low) {
            return rangeSumBST(root->right, low, high);
        }
        return root->val + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
    }
};
```

**复杂度分析**

- 时间复杂度：O*(*n)，其中n是二叉搜索树的节点数。
- 空间复杂度：O*(*n)，空间复杂度主要取决于栈空间的开销。



### 2021.4.30 哈希表

> LeetCode - 137 只出现一次的数字II   https://leetcode-cn.com/problems/single-number-ii/

```c++
int singleNumber(vector<int>& nums) {
	unordered_map<int, int> freq;
	for (int num : nums) {
		++freq[num];
	}
	int ans;
	for (auto [num, occ] : freq) {
		if (occ == 1) {
			ans = num;
			break;
		}
	}
	return ans;
}
```



### 2021.5.1 哈希表&dfs

> LeetCode - 690 员工的重要性  https://leetcode-cn.com/problems/employee-importance/

```c++
unordered_map<int, Employee*> mp; //定义哈希表，存储每一个员工及其id

int dfs(int id) {
    Employee* employee = mp[id];
    int total = employee->importance;
    //把每一个下属的重要度都加到total里
    for (int i : employee->subordinates) {
        total += dfs(i);
    }
    return total;
}


int getImportance(vector<Employee*> employees, int id) {
    //将employees读取到哈希表里
    for (auto& employee : employees) {
        mp[employee->id] = employee;
    }
    return dfs(id);
}
```



### 2021.5.10 dfs&二叉树叶子

> LeetCode - 872 叶子相似的树 https://leetcode-cn.com/problems/leaf-similar-trees/

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void dfs(TreeNode* root, vector<int>& vec){
        if(!root->left && !root->right){
            vec.push_back(root->val); //检测到叶节点，加入到数组里
        }
        else{  //不是叶节点
            if(root->left)  dfs(root->left, vec); //左子树下面还有叶节点，往下搜索
            if(root->right) dfs(root->right, vec); //右子树下面还有叶节点，往下搜索
        }
    }

    bool leafSimilar(TreeNode* root1, TreeNode* root2) {
        vector<int> vec1;
        if(root1) dfs(root1, vec1);

        vector<int> vec2;
        if(root2) dfs(root2, vec2);

        return vec1 == vec2;
    }
};
```



### 2021.5.17 二叉树&dfs+bfs

> LeetCode - 993 二叉树的堂兄弟节点 https://leetcode-cn.com/problems/cousins-in-binary-tree/

1.dfs 主要是利用dfs找到x和y,再比较x,y的深度和其父节点是否相同

```c++
class Solution {
private:
	int x_depth = 0;
	int y_depth = 0;
	TreeNode* x_father;
	TreeNode* y_father;
public:
    bool isCousins(TreeNode* root, int x, int y) {
        dfs(root,x,y,0);
        return x_depth == y_depth && x_father != y_father;
    }
	void dfs(TreeNode* root, int x, int y, int depth) {
		if(root == nullptr) return ;
		if(root->left != nullptr) {
			if(root->left->val == x) {
				x_depth = depth+1;
				x_father = root;
			}
			if(root->left->val == y) {
				y_depth = depth+1;
				y_father = root;
			}
			dfs(root->left,x,y,depth+1);
		}
		if(root->right != nullptr) {
			if(root->right->val == x) {
				x_depth = depth+1;
				x_father = root;
			}
			if(root->right->val == y) {
				y_depth = depth+1;
				y_father = root;
			}
			dfs(root->right,x,y,depth+1);
		}
	}
};
```

2.bfs 

我们使用的是层序遍历，如果每次遍历一层，那么这一层的元素的深度是相同的。

在确定深度上面，BFS一直很可以的。

**因此我们在每一层，看看是否有出现 x 和 y，其中分为以下三种情况：**

1.  x 和 y 都没出现 → 那只能往下一层继续找了

2.  x 和 y 只出现一个 → 两个元素的深度不同，不可能为兄弟，返回false

3.  x 和 y 都出现了，好耶，但是还不够好

   - `x` 和 `y` 父节点相同 → 不是堂兄弟，是亲兄弟，不行，返回`false`
   - `x` 和 `y` 父节点不同 → 满足题目条件了，好耶，返回`true`	

   众所周知，BFS需要用到队列，那么我们应该如何设计队列的数据类型？
   在这里，我采用了 **pair<TreeNode*, TreeNode*>（其实pair<TreeNode*, int>也可以），**

   其中pair中，第一个元素记录指向当前结点的指针，第二个元素记录指向当前结点的父结点的指针，这样就可以完美应对上面所说的三种情况了。

```c++
class Solution {
public:
    using PTT = pair<TreeNode*, TreeNode*>;
    bool isCousins(TreeNode* root, int x, int y) {
        // 使用队列q来进行bfs
        // 其中pair中，p.first 记录当前结点的指针，p.second 记录当前结点的父结点的指针
        queue<PTT> q;
        q.push({root, nullptr});
        while(!q.empty()) {
            int n = q.size();
            vector<TreeNode*> rec_parent;
            for(int i = 0; i < n; i++) {
                auto [cur, parent] = q.front(); q.pop();
                if(cur->val == x || cur->val == y)
                    rec_parent.push_back(parent);
                if(cur->left) q.push({cur->left, cur});
                if(cur->right) q.push({cur->right, cur});
            }
            // `x` 和 `y` 都没出现
            if(rec_parent.size() == 0)
                continue;
            // `x` 和 `y` 只出现一个
            else if(rec_parent.size() == 1)
                return false;
            // `x` 和 `y` 都出现了
            else if(rec_parent.size() == 2)
                // `x` 和 `y` 父节点 相同/不相同 ？
                return rec_parent[0] != rec_parent[1];
        }
        return false;
    }
};
```



### 2021.5.21 dp -- 最长公共子串

> LeetCode 1035 不相交的线 https://leetcode-cn.com/problems/uncrossed-lines/

![image-20210521111506583](C:\Users\86188\AppData\Roaming\Typora\typora-user-images\image-20210521111506583.png)

```c++
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size(), n = nums2.size();
        int dp[m+1][n+1];
        memset(dp, 0, sizeof(dp));
        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                if(nums1[i-1] == nums2[j-1])
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
        return dp[m][n];    
    }
};
```



### 2021.5.27 汉明距离

> LeetCode - 461 汉明距离 https://leetcode-cn.com/problems/hamming-distance/

两个整数之间的[汉明距离](https://baike.baidu.com/item/汉明距离)指的是这两个数字对应二进制位不同的位置的数目。

**1、直接使用内置计数功能**

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        return __builtin_popcount(x ^ y);
    }
};
```

**2、移位统计1的数量**

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int s = x ^ y, ret = 0;
        while (s) {
            ret += s & 1;
            s >>= 1;
        }
        return ret;
    }
};
```

**3、Brian Kernighan 算法**

跳过所有的0，直接对1进行计数：令f(x) = x & (x-1) 就会删除x最右侧的1

<img src="https://assets.leetcode-cn.com/solution-static/461/3.png" alt="fig3" style="zoom: 25%;" />

```c++
class Solution {
public:
    int hammingDistance(int x, int y) {
        int s = x ^ y, ret = 0;
        while (s) {
            s &= s - 1;
            ret++;
        }
        return ret;
    }
};
```



### 2021.5.28 汉明距离总和

> LeetCode - 477 汉明距离总和 https://leetcode-cn.com/problems/total-hamming-distance/

在计算汉明距离时，我们考虑的是同一比特位上的值是否不同，而不同比特位之间是互不影响的。

对于数组 ***nums*** 中的某个元素 ***val***，若其二进制的第 ***i*** 位为 1，我们只需统计 ***nums*** 中有多少元素的第 ***i*** 位为 0，即计算出了 ***val***与其他元素在第 ***i*** 位上的汉明距离之和。

具体地，若长度为 ***n*** 的数组 ***nums*** 的所有元素二进制的第 ***i*** 位共有 ***c*** 个 ***1***，***n-c*** 个 ***0***，则些元素在二进制的第 ***i*** 位上的汉明距离之和为

$$
c*(n-c)
$$

```c++
class Solution {
public:
    int totalHammingDistance(vector<int> &nums) {
        int ans = 0, n = nums.size();
        for (int i = 0; i < 30; ++i) {
            int c = 0;
            for (int val : nums) {
                c += (val >> i) & 1; //统计这一位上1的个数
            }
            ans += c * (n - c); //加上第i位上的汉明距离和
        }
        return ans;
    }
};
```



### 2021.5.30 2的幂

> LeetCode - 231 2的幂   https://leetcode-cn.com/problems/power-of-two/
>
> 一个数 *n* 是 2 的幂，当且仅当 *n* 是正整数，并且 *n* 的二进制表示中仅包含 1 个 1。
>
> n & (n - 1) 指的是删掉最右侧的一个1

```c++
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }
};
```



### 2021.5.31 4的幂

> LeetCode - 342 4的幂   https://leetcode-cn.com/problems/power-of-four/

在2的幂的基础上，要保证这个数的二进制数只有一个1且这个1在奇数位上。

构造一个mask，使其与n合取，来检验是否所有的1都在奇数位上。

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0b10101010101010101010101010101010) == 0;
    }
};
```

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && (n & 0xaaaaaaaa) == 0;
    }
};
```

另解：***利用以下性质***    

4的幂除以3余数一定为1，2的幂且不是4的幂除以3的余数一定为2

```c++
class Solution {
public:
    bool isPowerOfFour(int n) {
        return n > 0 && (n & (n - 1)) == 0 && n % 3 == 1;
    }
};
```



### 2021.6.1 前缀和&儿童节吃糖果

> LeetCode - 1744 吃最喜欢的糖果 https://leetcode-cn.com/problems/can-you-eat-your-favorite-candy-on-your-favorite-day/

```c++
class Solution {
public:
    vector<bool> canEat(vector<int>& candiesCount, vector<vector<int>>& queries) {
        int n = candiesCount.size();
        int m = queries.size();

        //前缀和
        vector<long> preSum(n);
        preSum[0] = candiesCount[0];
        for(int i=1; i<n; ++i) preSum[i] = preSum[i-1] + candiesCount[i];

        vector<bool> ans(m);
        for(int i=0; i<m; ++i)
        {
            int favoriteType = queries[i][0];
            long favoriteDay = queries[i][1];
            long dailyCap = queries[i][2];
            long minCandy = favoriteDay + 1;
            long maxCandy = (favoriteDay + 1) * dailyCap;
            // 理想区间是[day+1, (day+1)*dailyCap]
            if(favoriteType == 0 && preSum[favoriteType] >= minCandy) ans[i] = true; //type=0的特殊情况
            else if(favoriteType!=0 && preSum[favoriteType]>=minCandy && preSum[favoriteType-1]<maxCandy) ans[i] = true;
        }

        return ans;
    }
};
```



### 2021.6.2 前缀和&哈希表

> LeetCode - 523 连续的子数组和 https://leetcode-cn.com/problems/continuous-subarray-sum/

```c++
class Solution {
public:
    bool checkSubarraySum(vector<int>& nums, int k) {
        int m = nums.size();
        if (m < 2) {
            return false;
        }
        unordered_map<int, int> mp; //创建哈希表，记录每个余数出现的最小下标
        mp[0] = -1;
        int remainder = 0; //记录前缀和的余数
        for (int i = 0; i < m; i++) {
            remainder = (remainder + nums[i]) % k;
            if (mp.count(remainder)) {
                //此余数已经在哈希表里
                int prevIndex = mp[remainder]; //记录已经在哈希表里面的余数的下标
                if (i - prevIndex >= 2) {
                    return true; //下标差大于2，满足题目要求
                }
            } else {
                mp[remainder] = i; //此余数还未出现在哈希表中，加进去
            }
        }
        return false;
    }
};
```



### 2021.6.3 前缀和&哈希表

> LeetCode - 525 连续数组 https://leetcode-cn.com/problems/contiguous-array/

```c++
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        int maxLength = 0; //记录最大子数组长度
        unordered_map<int,int> mp; 
        mp[0] = -1;

        int counter = 0; //用于计数
        for(int i=0; i<nums.size(); ++i){
            if(nums[i]==1) counter++; //数组遍历到1，计数器+1
            else counter--; //数组遍历到0，计数器-1

            if(mp.count(counter)){
                //此数已经在哈希表中
                int prevIndex = mp[counter]; //记录上一次出现这个数的下标
                maxLength = max(maxLength, i-prevIndex); //取大
            }
            else mp[counter] = i; //这个数不在哈希表中，加进去
        }
        return maxLength;
    }
};
```



### 2021.6.4 相交链表

> LeetCode - 160 相交链表 https://leetcode-cn.com/problems/intersection-of-two-linked-lists/

一、哈希集合(自己写的)

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if(!headA || !headB) return nullptr;
        unordered_set<ListNode*> st;
        ListNode *curr = headA;
        while(curr){
            st.insert(curr);
            curr=curr->next;
        }
        curr=headB;
        while(curr){
            if(st.count(curr))return curr;
            curr=curr->next;
        }
        return nullptr;
    }
};
```

***复杂度分析***

时间复杂度：O(m+n)，其中 m 和 n 是分别是链表A 和 B 的长度。需要遍历两个链表各一次。

空间复杂度：O(m)，其中 m 是链表 headA 的长度。需要使用哈希集合存储链表 headA 中的全部节点。



二、双指针 O(1) 难想到

```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *pA = headA, *pB = headB;
        while (pA != pB) {
            pA = pA == nullptr ? headB : pA->next;
            pB = pB == nullptr ? headA : pB->next;
        }
        return pA;
    }
};
```



### 2021.6.6 动态规划

> LeetCode - 474 一和零 https://leetcode-cn.com/problems/ones-and-zeroes/

```c++
class Solution {
public:
    int countChar(string str,char ch){
        int cnt = 0;
        for(char c:str){
            if(c==ch) cnt++;
        }
        return cnt;
    }

    int findMaxForm(vector<string>& strs, int m, int n) {
        int length = strs.size();
        //定义三维数组，dp[i][j][k]表示前i个字符串中用j个0和k和1子集的最大大小
        //最终答案应该是dp[length][m][n]
        vector<vector<vector<int>>> dp(length+1, vector<vector<int>>(m+1,vector<int>(n+1)));
        for(int i=1; i<=length; ++i){
            int zeroNum = countChar(strs[i-1],'0');
            int oneNum = countChar(strs[i-1],'1');
            for(int j=0; j<=m; ++j){
                for(int k=0; k<=n; ++k){
                    dp[i][j][k] = dp[i-1][j][k];
                    if(zeroNum<=j && oneNum<=k){
                        dp[i][j][k] = max(dp[i][j][k], dp[i-1][j-zeroNum][k-oneNum]+1);
                    }
                }
            }
        }
        return dp[length][m][n];
    }
};
```



### 2021.6.7 动态规划

> LeetCode - 494 目标和  https://leetcode-cn.com/problems/target-sum/

```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum = 0;
        for(int num:nums) sum += num;
        int diff = sum - target; //负数的个数为 (sum - target) / 2
        if(diff<0 || diff%2!=0) return 0; //不满足要求，直接返回0
        int n = nums.size();
        int neg = diff/2;
        vector<vector<int>> dp(n+1,vector<int>(neg+1)); //dp[i][j]表示在前i个数中找到和为j的方案数
        dp[0][0] = 1; //边界条件 dp[0][0]=1, dp[0][j] = 0(j!=0)
        for(int i=1; i<=n; ++i){
            int num = nums[i-1];
            for(int j=0; j<=neg; ++j){
                if(num>j) dp[i][j] = dp[i-1][j]; //当前num已经大于j，不能选
                else if(num<=j) dp[i][j] = dp[i-1][j] + dp[i-1][j-num];
            }
        }
        return dp[n][neg];
     }
};
```



### 2021.6.9 动态规划

> LeetCode 879 - 盈利计划**(阿里3.8面试原题)**    https://leetcode-cn.com/problems/profitable-schemes/

```c++
class Solution {
public:
    int profitableSchemes(int n, int minProfit, vector<int>& group, vector<int>& profit) {
        int len = group.size(), MOD = (int)1e9 + 7;
        //dp[i][j][k]表示前i个工作中(不一定全选)，选择j个员工，并保持至少k利润的方案数
        //最终答案是 dp[len][j][minProfit] j从0~n遍历
        vector<vector<vector<int>>> dp(len+1, vector<vector<int>>(n+1,vector<int>(minProfit+1)));
        dp[0][0][0] = 1;//边界条件
        for(int i=1; i<=len; ++i){   
            int members = group[i-1];
            int earn = profit[i-1];
            for(int j=0; j<=n; ++j){
                for(int k=0; k<=minProfit; ++k){
                    if(members>j){
                        //人数过多，这工作不能选
                        dp[i][j][k] = dp[i-1][j][k];
                    }else{
                        //这工作可以选
                        //使用max(0,k-earn)是因为题目要求利润至少为k而不是恰好为k
                        dp[i][j][k] = (dp[i-1][j][k] + dp[i-1][j-members][max(0,k-earn)]) % MOD;
                    }
                }
            }
        }
        int sum = 0;
        for(int j=0; j<=n; ++j){
            sum = (sum+dp[len][j][minProfit])%MOD;
        }
        return sum;
    }
};
```



### 2021.6.10 动态规划

> LeetCode - 518 零钱兑换Ⅱ      https://leetcode-cn.com/problems/coin-change-2/

```c++
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        vector<int> dp(amount+1); //dp[i]代表总金额为i的选择方案数
        dp[0] = 1;//边界条件，什么都不选才能使总金额为0
        for(int& coin:coins){
            for(int i=coin; i<=amount; ++i){
                dp[i] += dp[i-coin];
            }
        }
        return dp[amount];
    }
};
```



### 2021.6.11 动态规划

> LeetCode - 279 完全平方数  https://leetcode-cn.com/problems/perfect-squares/

```c++
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n+1); //dp[i]表示组成和为i的完全平方数的个数最少
        dp[0] = 0; //组成0的个数为0
        for(int i=1; i<=n; ++i){
            int minn = INT_MAX; //记录所需最少完全平方数的个数(不算i在内)
            for(int j=1; j*j<=i; ++j){
                minn = min(minn, dp[i-j*j]); //遍历j找到最小的dp[i-j*j]
            }
            dp[i] = minn+1; //+1是因为还要加上i这个数
        }
        return dp[n];
    }
};
```



### 2021.6.22 回溯+剪枝

> LeetCode - 剑指Offer 38.字符串的排列 https://leetcode-cn.com/problems/zi-fu-chuan-de-pai-lie-lcof/

```c++
class Solution {
public:
    vector<string> permutation(string s) {
        dfs(s,0);
        return res;
    }
private:
    vector<string> res; 
    void dfs(string s, int x){
        //已经固定完所有字符
        if(x==s.size()-1){
            res.push_back(s); //添加到结果
            return;
        }

        //还未固定完所有字符
        set<int> st; //用于储存已经固定的字符
        for(int i=x; i<s.size(); ++i){
            if(st.find(s[i]) != st.end()) continue; //这个字符已经在这一位固定过，重复，故剪枝
            st.insert(s[i]); 
            swap(s[i],s[x]); //将s[i]固定到x位
            dfs(s,x+1); //继续往下固定x+1位字符
        }
    }
};
```



### 2021.6.23 哈希表(Hard)

> LeetCode - 149 直线上最多的点数 https://leetcode-cn.com/problems/max-points-on-a-line/

```c++
class Solution {
public:
    int maxPoints(vector<vector<int>>& points) {
        int n = points.size();
        //小于3个点，必在一条直线上
        if(n<3){
            return n;
        }

        int res = 2;//至少有2个点在一条直线上
        for(int i=0; i<n; ++i){
            unordered_map<double,int> mp; //储存每个斜率的计数
            for(int j=0; j<n; ++j){
                long long dx = points[j][0] - points[i][0];
                long long dy = points[j][1] - points[i][1];
                double slope = 1.0*dy/dx;
                if(mp.count(slope)){
                    mp[slope]++; //该斜率已在哈希表中，计数+1
                }else{
                    mp[slope]=2; //该斜率还未在哈希表中，初始化为2
                }
                res = max(res,mp[slope]);
            } 
        }
        return res;
    }
};
```



### 2021.7.3 字符频率排序

> LeetCode - 451 根据字符出现频率排序 https://leetcode-cn.com/problems/sort-characters-by-frequency/

1、哈希表+vector+sort配合使用

```c++
class Solution {
public:
    string frequencySort(string s) {
        unordered_map<char,int> mp;
        for(char c:s){
            mp[c]++;
        }
        vector<pair<char,int>> vec(mp.begin(),mp.end());
        sort(vec.begin(),vec.end(),[](pair<char,int> a ,pair<char,int> b)
        {
            return a.second > b.second;
        });
        string res;
        for(auto it:vec){
            for(int i=0; i<it.second; ++i){
                res+=it.first;
            }
        }
        return res;
    }
};
```



### 2021.7.11 二分

> LeetCode - 274 H指数 https://leetcode-cn.com/problems/h-index/

```c++
class Solution {
public:
    int hIndex(vector<int>& citations) {
        //二分查找，确定h
        int left = 0; //h最小是0
        int right = citations.size(); //上限
        while(left<=right){
            int mid = left + (right-left)/2;
            //检验
            int cnt = 0;
            for(auto& num:citations){
                if(num>=mid) cnt++;
            }
            if(cnt>=mid){
                left = mid + 1;
            }else {
                right = mid - 1;
            }
        }
        return right; //注意，这里一定return的是else里面的
    }
};
```



### 2021.7.14 二分&lower_bound()

> LeetCode - 1818 绝对值差和 https://leetcode-cn.com/problems/minimum-absolute-sum-difference/

```c++
class Solution {
public:
    int minAbsoluteSumDiff(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size();
        long long sum = 0;
        for(int i=0; i<n; ++i){
            sum += abs(nums1[i]-nums2[i]); //统计替换前的绝对差值和
        }
        long long ans = sum;
        vector<int> temp(nums1); //复制一份nums1，排序
        sort(temp.begin(),temp.end());
        for(int i=0; i<n; ++i){
            auto it = lower_bound(temp.begin(),temp.end(),nums2[i]);//找到第一个大于等于nums2[i]的下标
            if(it!=temp.end()){
                //找得到比nums2[i]大的数字
                ans = min(ans, sum - abs(nums1[i]-nums2[i]) + abs(*it-nums2[i]));
            }
            if(it!=temp.begin()){
                //两边都要检验
                ans = min(ans, sum - abs(nums1[i]-nums2[i]) + abs(*(--it)-nums2[i]));
            }
        }
        return ans%(int)(1e9+7);
    }
};
```



### 2021.7.15 count&bound&二分

> 剑指Offer 53 - 在排序数组中查找数字Ⅰhttps://leetcode-cn.com/problems/zai-pai-xu-shu-zu-zhong-cha-zhao-shu-zi-lcof/

1、利用count()

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return count(nums.begin(),nums.end(),target);
    }
};
```

2、利用lower_bound()和upper_bound()

```c++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        return upper_bound(nums.begin(),nums.end(),target) - lower_bound(nums.begin(),nums.end(),target);
    }
};
```

3、二分

<img src="https://pic.leetcode-cn.com/b4521d9ba346cad9e382017d1abd1db2304b4521d4f2d839c32d0ecff17a9c0d-Picture1.png" alt="Picture1.png" style="zoom:40%;" />

```c++
class Solution {
public:
    int BinarySearch(vector<int>& nums, int target){
        int left = 0, right = nums.size()-1;
        while(left<=right){
            int mid = left + (right - left)/2;
            if(nums[mid]<=target)left = mid + 1;
            else right = mid - 1;
        }
        return right;
    }

    int search(vector<int>& nums, int target) {
        return BinarySearch(nums,target) - BinarySearch(nums,target-1);
    }
};
```





















