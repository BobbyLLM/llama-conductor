const ACTIONS = {
  "moa-summarize": {
    title: "Summarize",
    prompt: "Summarize the following text using precise and concise language. Use headers and bulleted lists where appropriate.\n\n"
  },
  "moa-translate": {
    title: "Translate",
    prompt: "Translate the following text into English. If it is already in English, translate it into French. Preserve formatting.\n\n"
  },
  "moa-sentiment": {
    title: "Analyze Sentiment",
    prompt: "Analyze the sentiment of the following text. Identify the overall tone (positive, negative, neutral, mixed), key emotional signals, and confidence level. Be specific.\n\n"
  }
};

browser.contextMenus.create({
  id: "moa-parent",
  title: "MoA Chat",
  contexts: ["selection"]
});

for (const [id, action] of Object.entries(ACTIONS)) {
  browser.contextMenus.create({
    id: id,
    parentId: "moa-parent",
    title: action.title,
    contexts: ["selection"]
  });
}

browser.contextMenus.create({
  id: "moa-separator",
  parentId: "moa-parent",
  type: "separator",
  contexts: ["selection"]
});

browser.contextMenus.create({
  id: "moa-raw",
  parentId: "moa-parent",
  title: "Send to Chat",
  contexts: ["selection"]
});

browser.contextMenus.onClicked.addListener((info, tab) => {
  const selection = (info.selectionText || "").trim();
  if (!selection) return;

  let text;
  if (info.menuItemId === "moa-raw") {
    text = selection;
  } else {
    const action = ACTIONS[info.menuItemId];
    if (!action) return;
    text = action.prompt + selection;
  }

  browser.storage.local.set({ moaPendingText: text });
  browser.sidebarAction.open();
});